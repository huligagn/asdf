# =====
# 0 -> stdin
# 1 -> stdout
# 2 -> stderr

echo this is a test line > input.txt
exec 3<input.txt
cat <&3

exec 4>output.txt
echo newline >&4
cat output.txt

exec 5>>output.txt
echo appended line >&5
cat output.txt
