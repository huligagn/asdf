#! /bin/zsh
printf "%-5s %-10s %-4s\n" No. Name Mark
printf "%-5s %-10s %-4.2f\n" 1 Sarath 80.3456
printf "%-5s %-10s %-4.2f\n" 2 James 90.9023
printf "%-5s %-10s %-4.2f\n" 3 Jeff 77.236

echo -e "\e[1;31m This is red text \e[0m"
echo -e "\e[1;42m Green Backgroud \e[0m"

### Variable and Path Variable
pgrep gedit
# you may get 26157
cat /proc/26157/environ
cat /proc/26157/environ | tr '\0' '\n'

var="value"
echo $var
echo ${var}