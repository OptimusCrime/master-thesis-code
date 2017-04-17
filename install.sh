#!/bin/bash

# Fonts

echo -e ""
echo -e "\033[0;34mDownloading fonts\033[0m"
echo -e ""
curl -o fonts/arial.ttf https://raw.githubusercontent.com/OptimusCrime/master-thesis-resources/master/fonts/arial.ttf
curl -o fonts/arial-mono.ttf https://raw.githubusercontent.com/OptimusCrime/master-thesis-resources/master/fonts/arial-mono.ttf

# Wordlists

echo -e ""
echo -e "\033[0;34mDownloading wordlists\033[0m"
echo -e ""
curl -o config/wordlists//wordlist1.txt https://raw.githubusercontent.com/OptimusCrime/master-thesis-resources/master/wordlists/wordlist1.txt
curl -o config/wordlists//wordlist2.txt https://raw.githubusercontent.com/OptimusCrime/master-thesis-resources/master/wordlists/wordlist2.txt
curl -o config/wordlists//wordlist3.txt https://raw.githubusercontent.com/OptimusCrime/master-thesis-resources/master/wordlists/wordlist3.txt

# Finish

echo -e ""
echo -e "\033[0;34mInstallation complete\033[0m"
