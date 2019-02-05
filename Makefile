setup:
	source ~/miniconda2/bin/activate cs7641_1

linux:
	source /home/thgamble/apps/miniconda3/etc/profile.d/conda.sh

run-all: car mushroom

car:
	@make sep
	@echo "Car - Tree"
	python main.py -d car tree
	@make sep
	@echo "Car - Nearest"
	python main.py -d car nearest
	@make sep
	@echo "Car - Neural"
	python main.py -d car neural
	@make sep
	@echo "Car - Vector"
	python main.py -d car vector
	@make sep
	@echo "Car - Boost"
	python main.py -d car boost

mushroom:
	@make sep
	@echo "mushroom - Tree"
	python main.py -d mushroom tree
	@make sep
	@echo "mushroom - Nearest"
	python main.py -d mushroom nearest
	@make sep
	@echo "mushroom - Neural"
	python main.py -d mushroom neural
	@make sep
	@echo "mushroom - Vector"
	python main.py -d mushroom vector
	@make sep
	@echo "mushroom - Boost"
	python main.py -d mushroom boost

clean:
	rm -f results

help:
	cat Makefile

sep:
	@echo "---------------------------------------------------"
