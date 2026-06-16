odftex_2orderstep.x 	: odftex_2orderstep.o          
	  gfortran -O odftex_2orderstep.o -o odftex_2orderstep.x 
odftex_2orderstep.o	: odftex_2orderstep.f90       
	  gfortran -O  -c odftex_2orderstep.f90
