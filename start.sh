#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cache=${XDG_CACHE_HOME:-"$HOME/.cache"}/inkling2-mini
engine_repo=https://github.com/Maxrio9/TensorFold
release_api=https://api.github.com/repos/Maxrio9/TensorFold/releases
base_tag=inkling2-mini-exl3-v2
source_pin=${INKLING_TENSORFOLD_SOURCE_PIN:-}
model_repo=Maxrio/Inkling-2-Mini-exl3
drafter_repo=Maxrio/Inkling-2-Mini-DFlash2-FP8
drafter_upstream=incoai/Inkling-2-Mini-DFlash2
drafter_upstream_pin=4f9088ea66c21f2b99ed657d1941c306b03e84d2
original=("$@")
extra=()
card='' desktop='' branch='' engine_archive='' model_dir=''
long=false dry=false no_drafts=false vision_recipe=false

die() { printf 'start.sh: %s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'HELP'
Usage: ./start.sh [options] [-- SERVE_FLAGS...]
  --card 16|24|32|spark    Override GPU memory class
  --desktop|--no-desktop  Override display detection
  --long                  Use the 16 GB long-context recipe
  --vision                Use the matching vision recipe (24 GB and up)
  --no-drafts             Disable speculative decoding
  --branch BRANCH         Override the Hugging Face branch
  --engine-archive FILE   Use a local TensorFold release archive
  --model-dir DIR         Use a local model directory
  --dry-run               Print the plan without downloads or writes

Pass TensorFold flags after -- (./start.sh -- --port 8081).
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
        --vision) vision_recipe=true ;;
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
paged=true
if ((${capability/./} < 120)); then
    paged=false
    export TF_INKLING2_DRAFT_FP8_LAYERS=0 TF_INKLING2_DRAFT_FP8_HEAD=0 TF_INKLING2_LAZY_HEAD=0
fi

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
    16) recipe=16gb
        export TENSORFOLD_MEMORY_RESERVE_GIB=${TENSORFOLD_MEMORY_RESERVE_GIB:-0.5} ;;
    24|32) recipe=${card}gb ;;
    spark) recipe=dgx-spark ;;
    *) die '--card must be 16, 24, 32 or spark' ;;
esac
if $long; then
    [[ $card == 16 ]] || die '--long requires the 16 GB recipe'
    recipe=16gb-long
elif $desktop; then
    recipe+=-desktop
fi
if $vision_recipe; then
    [[ $card != 16 ]] || die '--vision requires a 24 GB or larger card class'
    recipe+=-vision
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
        --config|--context|--parallel|--kv-cache|--host|--port|--name|--drafter|--prompt-cache-gib|--vision-max-images|--vision-image-tokens|--vision-image-share)
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
        --vision)
            [[ $arg == --vision ]] || die '--vision does not take a value'
            overrides[--vision]=true ;;
        *) serve_flags+=("$arg") ;;
    esac
done
[[ -f $config ]] || die "Config file is missing: $config"
[[ ! $engine_archive || -f $engine_archive ]] || die 'Engine archive must be a file'
[[ ! $model_dir || -d $model_dir ]] || die 'Model override must be a directory'

settings=$(python3 - "$config" <<'PY'
import re
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    raw = stream.read()
config = tomllib.loads(raw.decode())
for key, default in (
    ("context", 0), ("parallel", 1), ("kv-cache", "fp8"), ("no-drafts", False),
    ("host", "127.0.0.1"), ("port", 8080), ("name", "inkling-2-mini"), ("drafter", ""),
    ("vision", False), ("vision-max-images", 4), ("vision-image-tokens", 4096),
    ("vision-image-share", "request"), ("prompt-cache-gib", ""),
):
    value = config.get(key, default)
    print(str(value).lower() if isinstance(value, bool) else value)
match = re.search(rb"^# Model branch: ([0-9.]+bpw)\.", raw, re.MULTILINE)
print(match[1].decode() if match else "")
print(str(b"# Start with TF_INKLING2_LAZY_HEAD=0 set." in raw.splitlines()).lower())
PY
)
mapfile -t settings <<< "$settings"
context=${overrides[--context]:-${settings[0]}}
[[ $context =~ ^(0|[1-9][0-9]*)$ ]] || die 'Context must be a nonnegative integer'
if ! $paged && { [[ $context == 0 ]] || [[ ! ${overrides[--context]:-} && $config == "$repo/configs/$recipe.toml" ]]; }; then
    die "Shared-pool presets require sm_120 or later. For legacy CUDA, pass -- --context N or a config with an explicit context."
fi
parallel=${overrides[--parallel]:-${settings[1]}}
kv=${overrides[--kv-cache]:-${settings[2]}}
host=${overrides[--host]:-${settings[4]}}
port=${overrides[--port]:-${settings[5]}}
name=${overrides[--name]:-${settings[6]}}
drafter=${overrides[--drafter]:-${settings[7]:-}}
vision=${overrides[--vision]:-${settings[8]}}
if [[ $vision == true && $card == 16 ]]; then die 'Vision requires a 24 GB or larger card class'; fi
max_images=${overrides[--vision-max-images]:-${settings[9]}}
image_tokens=${overrides[--vision-image-tokens]:-${settings[10]}}
image_share=${overrides[--vision-image-share]:-${settings[11]}}
prompt_cache=${overrides[--prompt-cache-gib]:-${settings[12]}}
if [[ ${settings[3]} == true ]]; then no_drafts=true; fi
if [[ $drafter == none ]]; then no_drafts=true; fi
if [[ ${settings[14]} == true ]]; then export TF_INKLING2_LAZY_HEAD=${TF_INKLING2_LAZY_HEAD:-0}; fi
if [[ ! $branch ]]; then
    branch=${settings[13]:-}
    if [[ ! $branch ]]; then
        if [[ $card == 16 ]]; then branch=3.3bpw; else branch=4.0bpw; fi
    fi
fi

printf 'GPU: %s, driver %s, %s, %s MiB total, %s MiB free\n' "$gpu" "$driver" "$sm" "$total" "$free"
printf 'Recipe: %s (card %s, desktop %s), model branch %s\n' "$config" "$card" "$desktop" "$branch"

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
    elif [[ $source_pin ]]; then printf 'Would build TensorFold for %s with Zig 0.17 and the CUDA toolkit.\n' "$sm"
    else printf 'A local rc1 engine archive is required until a compatible release or source pin is available.\n'
    fi
    if [[ $model_dir ]]; then printf 'Would use local model %s.\n' "$model_dir"
    else printf 'Would refresh %s (%s) with hf and verify SHA256SUMS.\n' "$model_repo" "$branch"
    fi
    if $paged; then
        printf 'Would plan %s streams at context %s with %s KV, then serve %s at http://%s:%s/v1/models.\n' \
            "$parallel" "$context" "$kv" "$name" "$url_host" "$port"
    else
        printf 'Would serve %s legacy CUDA streams at context %s with %s KV at http://%s:%s/v1/models.\n' \
            "$parallel" "$context" "$kv" "$url_host" "$port"
    fi
    if [[ $prompt_cache ]]; then
        printf 'Vision: %s. Prompt cache cap: %s GiB.\n' "$vision" "$prompt_cache"
    else
        printf 'Vision: %s. Prompt cache uses the engine default.\n' "$vision"
    fi
    if $no_drafts; then printf 'Speculative decoding would be disabled.\n'
    elif [[ $drafter ]]; then printf 'Would use drafter %s.\n' "$drafter"
    elif [[ $model_dir ]]; then printf 'Would use a verified DFlash2 bundle at %s/dflash2.\n' "$model_dir"
    elif ((${capability/./} >= 120)); then printf 'Would verify and install DFlash2 FP8, with pinned BF16 fallback.\n'
    else printf 'Would verify and install the pinned BF16 DFlash2 fallback.\n'
    fi
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
tag_prefix, minimum = base_tag.rsplit("v", 1)
pattern = re.compile(re.escape(tag_prefix) + r"v([1-9][0-9]*)")
archive = re.compile(r"tensorfold-[0-9]+\.[0-9]+\.[0-9]+-linux-" + re.escape(arch) + r"\.tar\.gz")
with open(path) as stream:
    releases = json.load(stream)
matches = []
for release in releases:
    tag = release["tag_name"]
    match = pattern.fullmatch(tag)
    if not match or int(match[1]) < int(minimum) or release["draft"] or release["prerelease"]:
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
    [[ $source_pin ]] || die 'No rc1-compatible release is available. Pass --engine-archive FILE for the local rc1 build.'
    [[ $source_pin =~ ^[[:xdigit:]]{40}$ ]] || die 'INKLING_TENSORFOLD_SOURCE_PIN must be a full commit hash'
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

managed_model=false
if [[ ! $model_dir ]]; then
    managed_model=true
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

verify_drafter() {
    # Accept only the expected bundle filenames, once each, before letting sha256sum open any path.
    awk '
        NF != 2 || length($1) != 64 || $1 ~ /[^[:xdigit:]]/ { exit 1 }
        {
            sub(/^\*/, "", $2)
            if ($2 !~ /^(config\.json|model\.safetensors|README\.md|LICENSE)$/ || seen[$2]++) exit 1
        }
        END { if (NR != 4) exit 1 }
    ' "$1/SHA256SUMS" &&
        (cd "$1" && sha256sum --check --quiet --strict SHA256SUMS)
}

install_drafter() {
    local origin='' status cache_key bundle file
    mkdir -p -- "$cache/drafters" "$stage/draft"
    if ((${capability/./} >= 120)); then
        origin=https://huggingface.co/$drafter_repo/resolve/main
        if ! status=$(fetch "$origin/SHA256SUMS" -o "$stage/draft/SHA256SUMS" -w '%{http_code}' 2>"$stage/draft-fetch-error"); then
            if [[ $status != 404 ]]; then
                cat "$stage/draft-fetch-error" >&2
                die "Could not check the DFlash2 FP8 package (HTTP $status)"
            fi
            origin=''
        fi
    fi
    if [[ ! $origin ]]; then
        origin=https://huggingface.co/$drafter_upstream/resolve/$drafter_upstream_pin
        cat > "$stage/draft/SHA256SUMS" <<'SUMS'
d094e435eb939e01fca6297f637c5af35e3da4b552f8e1ab0a181cd278a1cefa  config.json
217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee  model.safetensors
e809da60c54b754707657585ef479981925cfc9a14d3d4fe5223189b354a8a91  README.md
cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30  LICENSE
SUMS
        printf 'Using the pinned Inco AI BF16 DFlash2 checkpoint.\n'
    fi
    cache_key=$(sha256sum < "$stage/draft/SHA256SUMS")
    bundle=$cache/drafters/${cache_key%% *}
    if [[ ! -d $bundle ]]; then
        for file in config.json model.safetensors README.md LICENSE; do
            fetch "$origin/$file" -o "$stage/draft/$file" || die "Could not download DFlash2 $file"
        done
        verify_drafter "$stage/draft" || die 'DFlash2 checksum verification failed'
        mv -T -- "$stage/draft" "$bundle"
    else
        verify_drafter "$bundle" || die "DFlash2 checksum verification failed for $bundle"
        cmp -s "$stage/draft/SHA256SUMS" "$bundle/SHA256SUMS" || die "DFlash2 manifest changed in $bundle"
    fi
    ln -s -- "$bundle" "$stage/dflash2"
    mv -Tf -- "$stage/dflash2" "$model_dir/dflash2"
}

if ! $no_drafts && [[ ! $drafter ]]; then
    drafter=$model_dir/dflash2
    drafter_link=$(readlink -- "$drafter" || true)
    if $managed_model && { [[ ! -e $drafter && ! -L $drafter ]] || [[ $drafter_link == "$cache/drafters/"* ]]; }; then
        install_drafter
    else
        [[ -d $drafter ]] || die "DFlash2 bundle missing at $drafter. Supply -- --drafter DIR or use --no-drafts."
        verify_drafter "$drafter" || die "DFlash2 checksum verification failed in $drafter"
    fi
fi

model_flags=()
if $no_drafts; then
    model_flags+=(--no-drafts)
elif [[ $drafter ]]; then
    model_flags+=(--drafter "$drafter")
fi
if [[ $vision == true ]]; then
    model_flags+=(--vision --vision-max-images "$max_images" --vision-image-tokens "$image_tokens" --vision-image-share "$image_share")
fi
if [[ $prompt_cache ]]; then model_flags+=(--prompt-cache-gib "$prompt_cache"); fi

serve_flags+=("${model_flags[@]}")

verdict=$context
if $paged; then
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
    plan+=("${model_flags[@]}")
    selection=$(python3 "$repo/tools/recipes.py" --startup "$context" "${plan[@]}") ||
        die 'Could not determine a usable context'
    read -r verdict pool_tokens <<< "$selection"
    if [[ $context != 0 && $verdict != "$context" ]]; then
        printf 'Reducing context from %s to %s tokens to fit current free memory.\n' "$context" "$verdict"
    fi
    printf 'Planner admitted a %s-token KV pool with %s tokens per request.\n' "$pool_tokens" "$verdict"
fi
printf 'Model: %s\nOpen http://%s:%s/v1/models\n' "$name" "$url_host" "$port"
rm -rf -- "$stage"
trap - EXIT
exec 9>&-
exec "$binary" serve "$model_dir" --no-update-check --config "$config" --context "$verdict" --parallel "$parallel" \
    --kv-cache "$kv" --host "$host" --port "$port" --name "$name" "${serve_flags[@]}"
