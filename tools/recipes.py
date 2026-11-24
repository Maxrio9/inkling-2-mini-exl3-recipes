#!/usr/bin/env python3
"""Estimate an Inkling Q8 shared KV pool. All defaults are unmeasured priors."""
import argparse
import math

STEP = 4096
# Q8 payload: K and V * 8 heads * 128 dimensions * 1 byte, per visit.
Q8_BYTES_PER_VISIT = 2 * 8 * 128
PRIOR_GIB_PER_1K = 48 * Q8_BYTES_PER_VISIT * 1024 / 2**30


def pool(card, base, slope, reserves, native_max):
    if slope <= 0 or card <= 0 or base < 0 or reserves < 0 or native_max < 0:
        raise ValueError('invalid memory model')
    tokens = max(0, math.floor((card - base - reserves) / slope * 1024 / STEP) * STEP)
    return min(tokens, native_max // STEP * STEP)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--card', required=True, type=float, help='available GPU memory in GiB')
    p.add_argument('--base', required=True, type=float, help='fitted base in GiB, including loaded components')
    p.add_argument('--per-1k', type=float, default=PRIOR_GIB_PER_1K, help='fitted GiB per 1024 tokens; default is unmeasured payload prior')
    p.add_argument('--native-max', required=True, type=int, help='verified maximum shared pool tokens')
    p.add_argument('--desktop', action='store_true', help='reserve 1 GiB for display')
    p.add_argument('--slack', type=float, default=0.3)
    p.add_argument('--image-reserve', required=True, type=float, help='measured largest image working memory GiB, 0 for text only')
    p.add_argument('--extra-request', required=True, type=float, help='measured GiB per extra request')
    p.add_argument('--draft-reserve', type=float, default=0, help='draft incremental GiB only if not included in fitted base')
    a = p.parse_args()
    try:
        for n in (1, 4):
            reserve = a.slack + a.image_reserve + a.draft_reserve + (1 if a.desktop else 0) + (n-1)*a.extra_request
            print(f'{n} request(s): pool={pool(a.card, a.base, a.per_1k, reserve, a.native_max)} tokens; reserves={reserve:.3f} GiB')
    except ValueError as e:
        p.error(str(e))

if __name__ == '__main__':
    main()
