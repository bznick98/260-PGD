# Run Eval Code
cp /content/gdrive/MyDrive/data.zip /content/260-PGD
cd /content/260-PGD
unzip data
cd /content/260-PGD/data
mkdir /content/260-PGD/data/mmdetection
cp -r content/260-PGD/mmdetection/* /content/260-PGD/data/mmdetection
unzip eval_code
unzip images
mv images/* eval_code/select1000_new/
mv /content/260-PGD/hnm_pgd.py /content/260-PGD/hnm_utils.py /content/260-PGD/data/eval_code/
cd /content/260-PGD/data/eval_code
echo "Start Running hmn_pgd.py..."
python hnm_pgd.py