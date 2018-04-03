alias install="sudo apt-get install"

echo 'alias cmd="command seq"' >> ~./bashrc

alias rm='cp $@ ~/backup && rm $@'