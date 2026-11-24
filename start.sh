#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cache=${XDG_CACHE_HOME:-"$HOME/.cache"}/inkling2-mini
engine_repo=https://github.com/Maxrio9/TensorFold
release_api=https://api.github.com/repos/Maxrio9/TensorFold/releases
base_tag=inkling2-mini-exl3-v1
source_pin=664053f8f7cc9132985fc749bd3261519511b536
model_repo=Maxrio/Inkling-2-Mini-exl3
original=("$@")
extra=()
card='' desktop='' branch='' engine_archive='' model_dir=''
long=false dry=false no_drafts=false

die() { printf 'start.sh: %s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'HELP'
Usage: ./start.sh [options] [-- SERVE_FLAGS...]
  --card 16|24|32|spark    Override GPU memory class
  --desktop|--no-desktop  Override display detection
  --long                  Use the 16 GB long-context recipe
  --no-drafts             Disable DSpark
  --branch BRANCH         Override the Hugging Face branch
  --engine-archive FILE   Use a local TensorFold release archive
  --model-dir DIR         Use a local model directory
  --dry-run               Print the plan without downloads or writes
HELP
}

while (($#)); do
    case $1 in
        --card|--branch|--engine-archive|--model-dir)
            [[ ${2:-} && $2 != --* ]] || die "$1 requires a value"
            case $1 in
                --card) card=$2 ;;
                --branch) branch=$2 ;;
                --engine-archive) engine_archive=$(realpath -e -- "$2") ;;
                --model-dir) model_dir=$(realpath -e -- "$2") ;;
            esac
            shift ;;
        --desktop) desktop=true ;;
        --no-desktop) desktop=false ;;
        --long) long=true ;;
        --no-drafts) no_drafts=true ;;
        --dry-run) dry=true ;;
        -h|--help) usage; exit 0 ;;
        --) shift; extra=("$@"); break ;;
        *) die "unknown option $1 (place TensorFold flags after --)" ;;
    esac
    shift
done

[[ $(uname -s) == Linux ]] || die 'Linux is required'
command -v nvidia-smi >/dev/null || die 'Install the NVIDIA driver and make nvidia-smi available'
[[ ! ${CUDA_VISIBLE_DEVICES+x} || $CUDA_VISIBLE_DEVICES ]] || die 'CUDA_VISIBLE_DEVICES hides all GPUs'
gpu_query=('--query-gpu=uuid,name,driver_version,memory.total,memory.free,compute_cap,display_active' '--format=csv,noheader,nounits')
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
    gpu_query+=(-i "${CUDA_VISIBLE_DEVICES%%,*}")
fi
gpus=$(nvidia-smi "${gpu_query[@]}") || die 'nvidia-smi could not query the NVIDIA driver'
IFS=, read -r uuid gpu driver total free capability display <<< "${gpus%%$'\n'*}"
gpu=${gpu# } driver=${driver// /} total=${total// /} free=${free// /}
capability=${capability// /} display=${display// /}
[[ $uuid == GPU-* && $driver =~ ^[0-9]+\. && $capability =~ ^[0-9]+\.[0-9]+$ ]] ||
    die 'Could not read the GPU, driver version and compute capability'
sm=sm_${capability/./}
driver_info=$(nvidia-smi) || die 'Could not read the driver CUDA version'
cuda_version='CUDA (UMD )?Version: +([0-9]+)\.'
if ! [[ $driver_info =~ $cuda_version ]] || ((BASH_REMATCH[2] < 13)); then
    die "NVIDIA driver $driver must support CUDA 13 or later"
fi
export CUDA_VISIBLE_DEVICES=$uuid

unified=false
if [[ $sm == sm_121 ]]; then
    unified=true
    read -r total free < <(awk '/^MemTotal:/ {t=$2} /^MemAvailable:/ {a=$2} END {print int(t/1024), int(a/1024)}' /proc/meminfo)
fi
[[ $total =~ ^[0-9]+$ && $free =~ ^[0-9]+$ ]] || die "Could not read memory for $gpu"
if [[ ! $card ]]; then
    if [[ $sm == sm_121 ]]; then card=spark
    elif ((total >= 31 * 1024)); then card=32
    elif ((total >= 23 * 1024)); then card=24
    elif ((total >= 15 * 1024)); then card=16
    else die 'No automatic recipe fits GPUs below 16 GB'
    fi
fi
if [[ ! $desktop ]]; then
    desktop=false
    if [[ $display == Enabled || ${DISPLAY:-} || ${WAYLAND_DISPLAY:-} ]] ||
        pgrep -x 'Xorg|Xwayland' >/dev/null 2>&1; then desktop=true; fi
fi
case $card in
    16) recipe=16gb; branch=${branch:-3.3bpw}
        export TENSORFOLD_MEMORY_RESERVE_GIB=${TENSORFOLD_MEMORY_RESERVE_GIB:-0.5} ;;
    24|32) recipe=${card}gb; branch=${branch:-4.0bpw} ;;
    spark) recipe=dgx-spark; branch=${branch:-4.0bpw} ;;
    *) die '--card must be 16, 24, 32 or spark' ;;
esac
if $long; then
    [[ $card == 16 ]] || die '--long requires the 16 GB recipe'
    recipe=16gb-long
elif $desktop; then
    recipe+=-desktop
fi
arch=$(uname -m)
prebuilt=false
case "$arch/$sm" in
    x86_64/sm_86|x86_64/sm_89|x86_64/sm_120|aarch64/sm_121) prebuilt=true ;;
esac

config=$repo/configs/$recipe.toml
serve_flags=()
declare -A overrides=()
for ((i=0; i<${#extra[@]}; i++)); do
    arg=${extra[i]}
    flag=${arg%%=*}
    case $flag in
        --config|--context|--parallel|--kv-cache|--host|--port|--name|--drafter)
            if [[ $arg == *=* ]]; then
                value=${arg#*=}
            else
                i=$((i + 1))
                value=${extra[i]:-}
            fi
            [[ $value && $value != --* ]] || die "$flag requires a value"
            if [[ $flag == --config ]]; then config=$value; else overrides[$flag]=$value; fi
            ;;
        --no-drafts) no_drafts=true ;;
        *) serve_flags+=("$arg") ;;
    esac
done
[[ -f $config ]] || die "Config file is missing: $config"
[[ ! $engine_archive || -f $engine_archive ]] || die 'Engine archive must be a file'
[[ ! $model_dir || -d $model_dir ]] || die 'Model override must be a directory'

printf 'GPU: %s, driver %s, %s, %s MiB total, %s MiB free\n' "$gpu" "$driver" "$sm" "$total" "$free"
printf 'Recipe: %s (card %s, desktop %s), model branch %s\n' "$config" "$card" "$desktop" "$branch"

settings=$(python3 - "$config" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
for key, default in (
    ("context", 0), ("parallel", 1), ("kv-cache", "fp8"), ("no-drafts", False),
    ("host", "127.0.0.1"), ("port", 8080), ("name", "inkling-2-mini"), ("drafter", ""),
):
    value = config.get(key, default)
    print(str(value).lower() if isinstance(value, bool) else value)
PY
)
mapfile -t settings <<< "$settings"
context=${overrides[--context]:-${settings[0]}}
[[ $context =~ ^(0|[1-9][0-9]*)$ ]] || die 'Context must be a nonnegative integer'
parallel=${overrides[--parallel]:-${settings[1]}}
kv=${overrides[--kv-cache]:-${settings[2]}}
host=${overrides[--host]:-${settings[4]}}
port=${overrides[--port]:-${settings[5]}}
name=${overrides[--name]:-${settings[6]}}
drafter=${overrides[--drafter]:-${settings[7]:-}}
if [[ ${settings[3]} == true ]]; then no_drafts=true; fi

url_host=$host
case $host in 0.0.0.0|::) url_host=127.0.0.1 ;; *:*) url_host=[$host] ;; esac

if $dry; then
    if [[ $engine_archive && $model_dir ]]; then
        printf 'Would use local inputs and skip network updates.\n'
    else
        printf 'Would fast-forward the recipes checkout if clean and tracking an upstream.\n'
    fi
    if [[ $engine_archive ]]; then printf 'Would use engine archive %s.\n' "$engine_archive"
    elif $prebuilt; then printf 'Would check %s and newer releases for Linux %s and verify SHA256.\n' "$base_tag" "$arch"
    else printf 'Would build TensorFold for %s with Zig 0.17 and the CUDA toolkit.\n' "$sm"
    fi
    if [[ $model_dir ]]; then printf 'Would use local model %s.\n' "$model_dir"
    else printf 'Would refresh %s (%s) with hf and verify SHA256SUMS.\n' "$model_repo" "$branch"
    fi
    printf 'Would plan %s streams at context %s with %s KV, then serve %s at http://%s:%s/v1/models.\n' \
        "$parallel" "$context" "$kv" "$name" "$url_host" "$port"
    if $no_drafts || [[ $drafter == none ]]; then printf 'DSpark would be disabled.\n'; fi
    exit 0
fi

mkdir -p -- "$cache"
exec 9>"$cache/start.lock"
flock 9
stage=$(mktemp -d "$cache/.start.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT

if [[ ! $engine_archive || ! $model_dir ]] && [[ -e $repo/.git ]] && [[ ! ${INKLING_RECIPES_UPDATED:-} ]]; then
    if [[ $(git -C "$repo" status --porcelain) ]]; then
        printf 'Recipes checkout has local changes. Keeping the current checkout.\n' >&2
    elif git -C "$repo" rev-parse --verify --quiet '@{upstream}' >/dev/null; then
        before=$(git -C "$repo" rev-parse HEAD)
        if git -C "$repo" pull --ff-only; then
            if [[ $(git -C "$repo" rev-parse HEAD) != "$before" ]]; then
                rm -rf -- "$stage"
                exec 9>&-
                INKLING_RECIPES_UPDATED=1 exec "$repo/start.sh" "${original[@]}"
            fi
        else
            printf 'Recipes update failed. Using the current checkout.\n' >&2
        fi
    fi
fi

fetch() { curl -fLsS --proto '=https' --retry 2 --connect-timeout 10 "$@"; }
check_sha() {
    [[ $1 =~ ^[[:xdigit:]]{64}$ ]] || die "Invalid SHA256 for $2"
    printf '%s  %s\n' "$1" "$2" | sha256sum --check --status ||
        die "SHA256 mismatch for $2"
}
unpack_engine() {
    mkdir -p -- "$2"
    tar -xzf "$1" -C "$2" --strip-components=1 --no-same-owner --no-same-permissions
    [[ ! -e $2/HOST-ONLY ]] || die 'This archive has no GPU kernel images'
    [[ -x $2/bin/tensorfold-native ]] || die 'The archive has no executable bin/tensorfold-native'
    (cd "$2" && sha256sum --check --quiet --strict SHA256SUMS) ||
        die 'The extracted TensorFold archive failed verification'
}

binary=
if [[ $engine_archive ]]; then
    digest=$(sha256sum < "$engine_archive")
    install=$cache/engines/local-${digest%% *}
    if [[ ! -x $install/bin/tensorfold-native ]]; then
        unpack_engine "$engine_archive" "$stage/engine"
        mkdir -p -- "$cache/engines"
        mv -T -- "$stage/engine" "$install"
    fi
    binary=$install/bin/tensorfold-native
elif $prebuilt; then
    fetch "$release_api?per_page=100" -o "$stage/releases.json" ||
        die 'Could not check TensorFold releases. Use both local overrides for offline startup.'
    selection=$(python3 - "$stage/releases.json" "$arch" "$base_tag" <<'PY'
import json
import re
import sys

path, arch, base_tag = sys.argv[1:]
pattern = re.compile(re.escape(base_tag.rsplit("v", 1)[0]) + r"v([1-9][0-9]*)")
archive = re.compile(r"tensorfold-[0-9]+\.[0-9]+\.[0-9]+-linux-" + re.escape(arch) + r"\.tar\.gz")
with open(path) as stream:
    releases = json.load(stream)
matches = []
for release in releases:
    tag = release["tag_name"]
    match = pattern.fullmatch(tag)
    if not match or release["draft"] or release["prerelease"]:
        continue
    assets = {item["name"] for item in release["assets"]}
    if "SHA256SUMS" in assets:
        matches.extend((int(match[1]), tag, name) for name in assets if archive.fullmatch(name))
if matches:
    _, tag, name = max(matches)
    print(tag, name)
PY
)
    if [[ $selection ]]; then
        read -r tag asset <<< "$selection"
        download=$engine_repo/releases/download/$tag
        fetch "$download/SHA256SUMS" -o "$stage/release-sums"
        checksum=$(awk -v asset="$asset" '$2 == asset || $2 == "*" asset {print $1}' "$stage/release-sums")
        [[ $checksum =~ ^[[:xdigit:]]{64}$ ]] || die "No unique SHA256 for $asset"
        install=$cache/engines/$tag-${asset%.tar.gz}-$checksum
        if [[ ! -x $install/bin/tensorfold-native ]]; then
            fetch "$download/$asset" -o "$stage/$asset"
            check_sha "$checksum" "$stage/$asset"
            unpack_engine "$stage/$asset" "$stage/engine"
            mv -- "$stage/$asset" "$stage/engine/$asset"
            mkdir -p -- "$cache/engines"
            mv -T -- "$stage/engine" "$install"
        fi
        printf 'TensorFold release: %s\n' "$tag"
        binary=$install/bin/tensorfold-native
    fi
fi

if [[ ! $binary ]]; then
    nvcc=$(command -v nvcc) || die "No matching release archive. Install the CUDA toolkit (nvcc) to build for $sm."
    zig=$cache/zig-0.17.0-$arch/zig
    install=$cache/engines/source-$source_pin-$arch-$sm
    if [[ ! -x $install/native/bin/tensorfold-native ]]; then
        if [[ ! -x $zig ]]; then
            fetch https://ziglang.org/download/index.json -o "$stage/zig-index.json"
            zig_asset=$(python3 -c 'import json, sys
with open(sys.argv[1]) as stream:
    metadata = json.load(stream)["0.17.0"][sys.argv[2] + "-linux"]
print(metadata["tarball"], metadata["shasum"])
' "$stage/zig-index.json" "$arch")
            read -r zig_url zig_sha <<< "$zig_asset"
            fetch "$zig_url" -o "$stage/zig.tar.xz"
            check_sha "$zig_sha" "$stage/zig.tar.xz"
            mkdir -- "$stage/zig"
            tar -xJf "$stage/zig.tar.xz" -C "$stage/zig" --strip-components=1
            mv -T -- "$stage/zig" "${zig%/zig}"
        fi
        git init -q "$stage/source"
        git -C "$stage/source" fetch --depth 1 "$engine_repo.git" "$source_pin"
        git -C "$stage/source" checkout --detach FETCH_HEAD
        (
            cd "$stage/source"
            ZIG_GLOBAL_CACHE_DIR="$cache/zig-global" "$zig" build native -Dsm="${sm#sm_}" -Dnvcc="$nvcc" \
                -Dtarget="$arch-linux-gnu.2.28" -Doptimize=safe -Dstrip=true \
                --cache-dir "$cache/zig-cache" --prefix "$stage/source-engine"
        )
        mkdir -p -- "$cache/engines"
        mv -T -- "$stage/source-engine" "$install"
    fi
    binary=$install/native/bin/tensorfold-native
fi

if [[ ! $model_dir ]]; then
    hf=$(command -v hf || true)
    if [[ ! $hf ]]; then
        hf=$cache/hf/bin/hf
        if [[ ! -x $hf ]]; then
            if command -v uv >/dev/null; then
                uv venv "$cache/hf"
                uv pip install --python "$cache/hf/bin/python" huggingface_hub
            else
                python3 -m venv "$cache/hf"
                "$cache/hf/bin/python" -m pip install huggingface_hub
            fi
        fi
    fi
    branch_key=$(printf '%s' "$branch" | sha256sum)
    model_dir=$cache/models/${branch_key%% *}
    "$hf" download "$model_repo" --revision="$branch" --local-dir "$model_dir"
    (cd "$model_dir" && sha256sum --check --quiet --strict SHA256SUMS) ||
        die 'Downloaded model files failed SHA256SUMS verification'
fi

plan=("$binary" plan "$model_dir" --device-mib "$total" --parallel "$parallel" --kv-cache "$kv" --json)
if $unified; then
    free=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    plan+=(--unified)
else
    free=$(nvidia-smi -i "$uuid" --query-gpu=memory.free --format=csv,noheader,nounits)
    [[ $free =~ ^[0-9]+$ ]] || die 'Could not refresh free GPU memory'
    # nvidia-smi runs before the engine opens its CUDA context.
    free=$((free > 335 ? free - 335 : 0))
fi
if ((free > total)); then free=$total; fi
plan+=(--free-mib "$free")
if [[ ${TENSORFOLD_MEMORY_RESERVE_GIB:-} ]]; then plan+=(--reserve-gib "$TENSORFOLD_MEMORY_RESERVE_GIB"); fi
if [[ ${TENSORFOLD_CUDA_MEMORY_LIMIT_GB:-} ]]; then plan+=(--limit-gib "$TENSORFOLD_CUDA_MEMORY_LIMIT_GB"); fi
if $no_drafts || [[ $drafter == none ]]; then plan+=(--no-drafts); fi
# A valid planner refusal exits 1 and still yields a report with the maximum.
status=0
"${plan[@]}" > "$stage/plan.json" || status=$?
((status <= 1)) || die "TensorFold planner exited with status $status"
verdict=$(python3 - "$stage/plan.json" "$context" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    try:
        plan = json.load(stream)
    except json.JSONDecodeError:
        sys.exit("TensorFold did not return a JSON capacity report")
if "result" not in plan:
    sys.exit(plan.get("error_message", "TensorFold did not return a capacity report"))
maximum = plan["maximum"]
if not maximum:
    sys.exit(plan["result"]["message"] or "No usable context fits")
requested = int(sys.argv[2])
print(min(requested, maximum) if requested else maximum)
PY
) || die 'Could not determine a usable context'
if [[ $context != 0 && $verdict != "$context" ]]; then
    printf 'Reducing context from %s to %s tokens to fit current free memory.\n' "$context" "$verdict"
elif [[ $context == 0 ]]; then
    printf 'Planner admitted %s tokens per request.\n' "$verdict"
fi

if $no_drafts; then serve_flags+=(--no-drafts); fi
if [[ ${overrides[--drafter]:-} ]]; then serve_flags+=(--drafter "$drafter"); fi
printf 'Model: %s\nOpen http://%s:%s/v1/models\n' "$name" "$url_host" "$port"
rm -rf -- "$stage"
trap - EXIT
exec 9>&-
exec "$binary" serve "$model_dir" --no-update-check --config "$config" --context "$verdict" --parallel "$parallel" \
    --kv-cache "$kv" --host "$host" --port "$port" --name "$name" "${serve_flags[@]}"
