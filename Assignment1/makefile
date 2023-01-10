CC		= gcc
LD		= gcc
CFLAG		= -Wall
PROG_NAME	= Assignment_1

SRC_DIR		=./src
INC_DIR		=./inc
OBJ_DIR		=./build
BIN_DIR		=./bin
SRC_LIST	= $(wildcard $(SRC_DIR)/*.c)
OBJ_LIST	= $(wildcard $(OBJ_DIR)/*.o)


.PHONY : all clean $(PROG_NAME) compile

all: $(PROG_NAME)

compile:
	$(CC) $(SRC_LIST) -I $(INC_DIR) -c

copy: compile
	@mv $(notdir $(SRC_LIST:.c=.o)) $(OBJ_DIR)/

$(PROG_NAME): copy
	$(LD) $(OBJ_LIST) -lm -o $(BIN_DIR)/$@

clean:
	rm -f $(BIN_DIR)/$(PROG_NAME) $(OBJ_DIR)/*.o
