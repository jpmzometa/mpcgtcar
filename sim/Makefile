CC = gcc
FLAGS = -Os -Wall -Wstrict-prototypes -g
OPT = -O2 -funroll-loops
STD = -std=c90

all: lcmpc gtcar 

lcmpc: 
	cd ./cmpc ; make 

gtcar: main.c lcmpc
	$(CC) $(FLAGS) $(OPT) $(STD) -static main.c -Lcmpc -lcmpc -o gtcar

clean:
	rm -rf cmpc python __pycache__ gtcar
