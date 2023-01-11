#/usr/bin/env bash

# set -x
set -e

yaml_path="$1"
if [ -z "$yaml_path" ]; then
    echo "Inform yaml path"
    exit 1
fi

for i in "$yaml_path"/half_pool-*.yaml; do
    echo "$i"
    out_basename=${i##*/}
    out_path=${i%/*}
    out_csv="${out_path}/${out_basename}.csv"
    out_out="${out_path}/${out_basename}.out"
    printf "CSV path: %s\nOUT path: %s\n" "$out_csv" "$out_out"
    time python -u validation.py                        \
                -c "$i"                                 \
                -i ~/ninfa/datasets/data_classified_v6/     \
                -o "$out_csv" |& tee "$out_out"
done

