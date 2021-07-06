echo -n > all_unmarked_data.csv;
gsutil ls -R gs://parking_data_318120/ | grep 'jpg' | while read line; do
    echo $line','`echo $line | grep 'jpg' | awk -F / '{print $4}'` >> all_unmarked_data.csv;
done

FS="
"
classes=(`cat all_unmarked_data.csv | awk -F , '{print $NF}' | sort | uniq`)

echo -n > all_data.csv;
for i in "${classes[@]}"; do
    count_class=`cat all_unmarked_data.csv | grep ",$i\$" | wc -l`
    eighty=$(echo "$count_class/10*8" | bc)
    ten=$(echo "$count_class/10" | bc)
    class_lines=`grep $i all_unmarked_data.csv`
    if [[ $count_class -lt 10 ]]; then
        continue
    fi

    grep ",$i\$" all_unmarked_data.csv | sed -n "1,$(($eighty+1))p" | awk '{printf("TRAIN,%s\n", $0)}' >> all_data.csv
    grep ",$i\$" all_unmarked_data.csv | sed -n "$(($eighty+1)),$(($eighty+$ten+1))p" | awk '{printf("VALIDATION,%s\n", $0)}' >> all_data.csv
    grep ",$i\$" all_unmarked_data.csv | sed -n "$(($eighty+$ten+1)),$(($eighty+$ten+$ten+1))p" | awk '{printf("TEST,%s\n", $0)}' >> all_data.csv
done

