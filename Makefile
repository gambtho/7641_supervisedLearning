setup:
	source ${HOME}/miniconda2/bin/activate cs7641_1

run-all: car mushrooms

car: 
	make sep
	echo "Car - Tree"
	python main.py -d car tree
	make sep
	echo "Car - Nearest"
	python main.py -d car nearest
	make sep
	echo "Car - Neural"
	python main.py -d car neural
	make sep
	echo "Car - Vector"
	python main.py -d car vector
	make sep
	echo "Car - Boost"
	python main.py -d car boost

mushrooms:
	make sep
	echo "Mushrooms - Tree"
	python main.py -d mushrooms tree
	make sep
	echo "Mushrooms - Nearest"
	python main.py -d mushrooms nearest
	make sep
	echo "Mushrooms - Neural"
	python main.py -d mushrooms neural
	make sep
	echo "Mushrooms - Vector"
	python main.py -d mushrooms vector
	make sep
	echo "Mushrooms - Boost"
	python main.py -d mushrooms boost

clean:
	rm -f results

help:
	cat Makefile

sep:
	@echo "---------------------------------------------------"
