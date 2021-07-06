find ./parking_fotos_to_transfer/ -regex '\./parking_photos_raw/.*\.jpg' -print | sort | parallel -j `nproc` ./job.sh {}
