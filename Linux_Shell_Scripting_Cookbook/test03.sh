echo "This is a test text" > tmp.txt
echo "This is a test text 2" >> tmp.txt

cat tmp.txt

# =====
ls + > out.txt
ls + 2> out.txt
cmd 2>stderr.txt 1>stdout.txt

# stderr转成stdout，重定向到一个文件
cmd 2>&1 output.txt
cmd &> output.txt

cmd 2>/dev/null # black hole

# =====
command | tee FILE1 FILE2

cmd1 | cmd2 | cmd -
# e.g.
echo who is this | tee -