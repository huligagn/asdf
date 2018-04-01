#! /bin/zsh

### Variable practice
fruit=apple
count=5
echo "We have $count ${fruit}(s)"
echo ------------------------------------------------

### Path Variable
echo $PATH
echo $PATH | tr ':' '\n'

echo ------------------------------------------------
echo $HOME
echo $PWD
echo $USER
echo $UID
echo $SHELL


### the difference between '' and ""
echo ------------------------------------------------
var=hello
echo '$var'
echo "$var"
echo ------------------------------------------------