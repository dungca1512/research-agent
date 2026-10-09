#!/usr/bin/env bash
# security-gate — the single entry point every layer calls to scan for secrets.
# Wraps gitleaks so the scanner can be swapped in one place.
#
# Exit: 0 clean | 1 secrets found | 2 usage error | 3 scanner missing or failed (fail-closed)
set -u

SG_VERSION="2"
ZERO_RE='^0+$'

usage() {
  cat >&2 <<'EOF'
usage: security-gate <mode>
  staged    scan staged changes (pre-commit)
  push      scan commits being pushed; reads pre-push stdin; args: <remote> [url]
  tree      scan tracked + untracked-not-ignored files of the current repo
  history   scan the full git history (CI)
  deploy    tree + unpushed commits; a pass is cached for a few minutes
exit: 0 clean | 1 secrets found | 2 usage | 3 scanner missing or failed
EOF
  exit 2
}

# Directory that holds this script's project, following symlinks (~/.local/bin/security-gate).
resolve_root() {
  local src="${BASH_SOURCE[0]}" dir
  while [ -L "$src" ]; do
    dir="$(cd "$(dirname "$src")" && pwd)"
    src="$(readlink "$src")"
    case "$src" in /*) ;; *) src="$dir/$src" ;; esac
  done
  (cd "$(dirname "$src")/.." 2>/dev/null && pwd)
}

find_gitleaks() {
  if [ -n "${SECURITY_GATE_GITLEAKS+x}" ]; then
    [ -x "$SECURITY_GATE_GITLEAKS" ] && printf '%s\n' "$SECURITY_GATE_GITLEAKS"
    return
  fi
  command -v gitleaks 2>/dev/null && return 0
  local p
  for p in /opt/homebrew/bin/gitleaks /usr/local/bin/gitleaks; do
    [ -x "$p" ] && { printf '%s\n' "$p"; return 0; }
  done
  return 1
}

in_repo() { git rev-parse --is-inside-work-tree >/dev/null 2>&1; }
has_commits() { git rev-parse --verify -q HEAD >/dev/null 2>&1; }

# worst <a> <b>: 3 beats 1 beats 0
worst() {
  if [ "$1" -eq 3 ] || [ "$2" -eq 3 ]; then echo 3
  elif [ "$1" -ne 0 ] || [ "$2" -ne 0 ]; then echo 1
  else echo 0; fi
}

# Run gitleaks and map its exit code (--exit-code 2 separates "leaks" from "error").
# gitleaks:allow comments are never honoured: a one-line comment must not be able to silence the gate.
gl() {
  local rc
  "$GL" "$@" ${CFG_ARGS[@]+"${CFG_ARGS[@]}"} --ignore-gitleaks-allow \
    --verbose --redact --no-banner --no-color --exit-code 2 </dev/null 1>&2
  rc=$?
  case "$rc" in
    0) return 0 ;;
    2) echo "security-gate: SECRETS FOUND (values redacted above)" >&2; return 1 ;;
    *) echo "security-gate: scanner failed (gitleaks exit $rc)" >&2; return 3 ;;
  esac
}

# gitleaks always reads a .gitleaksignore from the directory it scans, whatever --gitleaks-ignore-path says.
# In trusted mode, scan the git dir instead of the work tree, so a repo-level ignore file is never read.
git_source() {
  if [ -n "$CFG_FILE" ]; then git rev-parse --absolute-git-dir; else echo .; fi
}

scan_range() {
  local src
  src="$(git_source)" || return 3
  gl git --log-opts="$1" "$src"
}

# Staged scans need the work tree, so a repo-level .gitleaksignore can still hide a finding here; the
# push, history and deploy scans do not read it, so such a commit is caught before it leaves the machine.
scan_staged() {
  in_repo || { echo "security-gate: not a git repository, nothing to scan" >&2; return 0; }
  gl git --pre-commit --staged
}

scan_history() {
  in_repo || { echo "security-gate: not a git repository" >&2; return 3; }
  has_commits || return 0
  local src
  src="$(git_source)" || return 3
  gl git "$src"
}

scan_unpushed() {
  in_repo || return 0
  has_commits || return 0
  local up
  if up="$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null)"; then
    scan_range "$up..HEAD"
  else
    scan_range "HEAD --not --remotes"
  fi
}

scan_push() {
  local remote="${1:-origin}" lref lsha rref rsha rc=0 r saw=0
  in_repo || return 0
  while read -r lref lsha rref rsha; do
    [ -n "$lsha" ] || continue
    saw=1
    if printf '%s' "$lsha" | grep -Eq "$ZERO_RE"; then continue; fi   # branch deletion
    if printf '%s' "$rsha" | grep -Eq "$ZERO_RE" || ! git cat-file -e "${rsha}^{commit}" 2>/dev/null; then
      scan_range "$lsha --not --remotes=$remote"
    else
      scan_range "$rsha..$lsha"
    fi
    r=$?
    rc="$(worst "$rc" "$r")"
  done
  if [ "$saw" -eq 0 ]; then scan_unpushed; rc=$?; fi
  return "$rc"
}

# NUL-separated list of the files a tree scan covers: tracked + untracked-not-ignored, readable regular
# files, no symlinks. Run from the repo top. The deploy cache key hashes exactly this list.
scan_list() {
  git ls-files -z --cached --others --exclude-standard | while IFS= read -r -d '' f; do
    if [ -f "$f" ] && [ ! -L "$f" ] && [ -r "$f" ]; then printf '%s\0' "$f"; fi
  done
}

# gitleaks dir ignores .gitignore, so scan a snapshot of the files in scan_list.
scan_tree() {
  local top tmp list rc
  if ! in_repo; then
    gl dir . --max-target-megabytes 5
    return
  fi
  top="$(git rev-parse --show-toplevel)" || return 3
  tmp="$(mktemp -d)" || return 3
  list="$tmp/files.list"
  ( cd "$top" && scan_list ) > "$list"
  mkdir "$tmp/tree"
  if ! tar -c -C "$top" --null -T "$list" -f - | tar -x -C "$tmp/tree"; then
    rm -rf "$tmp"
    echo "security-gate: could not snapshot the working tree" >&2
    return 3
  fi
  # Trusted mode: the repo's own ignore file is not honoured (see git_source).
  if [ -n "$CFG_FILE" ]; then rm -f "$tmp/tree/.gitleaksignore"; fi
  ( cd "$tmp/tree" && gl dir . --max-target-megabytes 5 )
  rc=$?
  rm -rf "$tmp"
  return "$rc"
}

# Hash of everything that decides a deploy-scan result: the scanner and its config, HEAD, all remote refs,
# and the path + content of every file scan_list returns. Any failure returns non-zero (the caller then
# neither reads nor writes the cache), so a partially computed key can never produce a cache hit.
cache_key() {
  local top tmp
  top="$(git rev-parse --show-toplevel)" || return 1
  tmp="$(mktemp -d)" || return 1
  (
    set -o pipefail
    cd "$top" || exit 1
    scan_list > "$tmp/list" || exit 1
    {
      echo "$SG_VERSION"
      "$GL" version || exit 1
      env | { grep '^GITLEAKS_' || true; } | LC_ALL=C sort
      echo "$top"
      git rev-parse --verify -q HEAD || true
      git rev-parse --verify -q '@{upstream}' || true
      git for-each-ref --format='%(objectname) %(refname)' refs/remotes
      if [ -n "$CFG_FILE" ]; then cat "$CFG_FILE" || exit 1; fi
      if [ -n "$CFG_IGNORE" ] && [ -f "$CFG_IGNORE" ]; then cat "$CFG_IGNORE" || exit 1; fi
      shasum -a 256 < "$tmp/list" || exit 1
      xargs -0 -r git hash-object --no-filters -- < "$tmp/list" || exit 1
    } | shasum -a 256 | cut -d' ' -f1
  )
  local rc=$?
  rm -rf "$tmp"
  return "$rc"
}

cache_dir_ok() { [ -d "$1" ] && [ ! -L "$1" ] && [ -O "$1" ]; }

cache_store() {
  local dir="$1" key="$2" stamp="$3" tmp
  mkdir -p "$dir" 2>/dev/null || return 0
  cache_dir_ok "$dir" || return 0
  chmod 700 "$dir" 2>/dev/null || return 0
  tmp="$dir/.tmp.$$"
  echo "$stamp" > "$tmp" && mv -f "$tmp" "$dir/$key"
  return 0
}

scan_deploy() {
  in_repo || { echo "security-gate: not a git repository, nothing to scan" >&2; return 0; }
  local dir="${SECURITY_GATE_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/security-gate}"
  local ttl="${SECURITY_GATE_CACHE_TTL:-300}" key key2 stamp now rc r
  now="$(date +%s)"
  key="$(cache_key)" || key=""   # no key: scan every time, never touch the cache
  if [ -n "$key" ] && cache_dir_ok "$dir" && [ -f "$dir/$key" ] && [ -O "$dir/$key" ]; then
    stamp="$(cat "$dir/$key" 2>/dev/null)"
    case "$stamp" in ''|*[!0-9]*) stamp=0 ;; esac
    if [ $((now - stamp)) -lt "$ttl" ]; then
      echo "security-gate: deploy gate passed (cached)" >&2
      return 0
    fi
  fi
  scan_tree; rc=$?
  scan_unpushed; r=$?
  rc="$(worst "$rc" "$r")"
  if [ "$rc" -eq 0 ]; then
    echo "security-gate: deploy gate passed" >&2
    # A pass describes the state it was computed for: cache it only if nothing changed during the scan.
    if [ -n "$key" ] && key2="$(cache_key)" && [ "$key2" = "$key" ]; then
      cache_store "$dir" "$key" "$now"
    fi
  fi
  return "$rc"
}

mode="${1:-}"
[ $# -gt 0 ] && shift
case "$mode" in
  version|--version) echo "security-gate $SG_VERSION"; exit 0 ;;
  staged|push|tree|history|deploy) ;;
  *) usage ;;
esac

GL="$(find_gitleaks)" || {
  echo "security-gate: gitleaks not found (brew install gitleaks) — blocking (fail-closed)" >&2
  exit 3
}
"$GL" version >/dev/null 2>&1 || {
  echo "security-gate: gitleaks at $GL is not runnable — blocking (fail-closed)" >&2
  exit 3
}

# Trusted mode: scan with a config and ignore file that live next to the gate, never the repo's own, so
# a repo (or an agent editing it) cannot allowlist its way past the scan. Without them (the script
# vendored into CI) the repo's own config applies, and review of the pull request is the control.
SG_ROOT="$(resolve_root)"
CFG_FILE=""
CFG_IGNORE=""
CFG_ARGS=()
if [ -n "${SECURITY_GATE_CONFIG+x}" ]; then
  [ -f "$SECURITY_GATE_CONFIG" ] || {
    echo "security-gate: config $SECURITY_GATE_CONFIG not found — blocking (fail-closed)" >&2
    exit 3
  }
  CFG_FILE="$SECURITY_GATE_CONFIG"
elif [ -n "$SG_ROOT" ] && [ -f "$SG_ROOT/gitleaks.trusted.toml" ]; then
  CFG_FILE="$SG_ROOT/gitleaks.trusted.toml"
fi
if [ -n "$CFG_FILE" ]; then
  CFG_IGNORE="$SG_ROOT/.gitleaksignore"
  CFG_ARGS=(--config "$CFG_FILE" --gitleaks-ignore-path "$SG_ROOT")
fi

case "$mode" in
  staged)  scan_staged ;;
  push)    scan_push "$@" ;;
  tree)    scan_tree ;;
  history) scan_history ;;
  deploy)  scan_deploy ;;
esac
exit $?
