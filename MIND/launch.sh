cd raw 
./download.sh mind .
./download.sh glove .
./download.sh feeds .

cd ../

cd preprocess
python news_glove_process.py --data mind
python news_popular.py --data mind
python user_process.py --data mind

python news_glove_process.py --data feeds --min_word_cnt 10
python news_popular.py --data feeds
python user_process.py --data feeds