reset;

################################
# Sets & Parameters
################################
set Time; 	#time steps for which we have results 

#parameter defined in dat sheet: 

param T_cond {Time}; 		#[°C] condensation temperature of the hp 
param T_evap {Time}; 		#[°C] evaporation temperature of the hp 
param T_ext {Time}; 		#[°C] external temperature 
param T_hp_5 {Time}; 		#[°C] temperature in 5, see flowsheet
param T_EPFL_out {Time}; 	#[°C] temperature return EPFL
param T_EPFL_in {Time}; 	#[°C] temperature supply EPFL

param T_lake_in := 7;
param T_lake_out := 3;


param Q_cond{Time}; 		#[kW] heat condensor
param Q_evap{Time}; 		#[kW] heat evaporator
param W_comp1{Time};		#[kW] power of compressor 1
param W_comp2{Time}; 	#[kW] power of compressor 1 

param W_hp{t in Time}:= W_comp1[t] + W_comp2[t];  		#[kW] total power consumed by hp 



#costing parameters (a gift from your great TAs, as you already should have done this exercise for part 2 and this already takes enough time ;-) 
param k1  				:= 2.2897;	#parameter for compressor cost function
param k2 			 	:= 1.3604; #parameter for compressor cost function
param k3  				:= -0.1027; #parameter for compressor cost function
param k1_HEX			:= 4.8306;
param k2_HEX	 	 	:= -0.8509;
param k3_HEX  			:= 0.3187;
param f_BM 				:=  2.7; 	# bare module factor for compressor (CS) 
param f_BM_HEX			:=  4.74; 	# bare module factor for HEX  
param ref_index 		:= 394.3; 	# CEPCI reference 2001 
param index 			:= 537.5 ;	# CEPCI 2016

param i 				:= 0.06 ; #interest rate
param n 				:= 20; #[y] life-time

param U_water_ref       := 0.75; #water-refrigerant global heat transfer coefficient (kW/m2.K)
param U_air_ref         := 0.049; #air-refrigerant global heat transfer coefficient (kW/m2.K)

#Extreme values
#################
# !!!!! Fill here: : take temporal index from maximal heat demand
param max_demand_index;
param min_demand_index;

#################
param T_cond_max := T_cond[max_demand_index];
param Q_cond_max := Q_cond[max_demand_index];
param T_evap_max := T_evap[max_demand_index];
param Q_evap_max := Q_evap[max_demand_index];
param T_EPFL_in_max := T_EPFL_in[max_demand_index];
param T_EPFL_out_max := T_EPFL_out[max_demand_index];
param W_comp1_max := W_comp1[max_demand_index];
param W_comp2_max := W_comp2[max_demand_index];
param W_hp_max := W_comp1_max+W_comp2_max;

param T_cond_min := T_cond[min_demand_index];
param Q_cond_min := Q_cond[min_demand_index];
param W_comp1_min := W_comp1[min_demand_index];
param W_comp2_min := W_comp2[min_demand_index];
param T_evap_min := T_evap[min_demand_index];
param Q_evap_min := Q_evap[min_demand_index];
param T_EPFL_in_min := T_EPFL_in[min_demand_index];
param T_EPFL_out_min := T_EPFL_out[min_demand_index];
param W_hp_min := W_comp1_min+W_comp2_min;

#################
# !!!!! Fill here: maximum mass flow according to previously defined parameters (Q=mcp*dT)
#################
param Mcp := Q_cond_max/(T_EPFL_in_max-T_EPFL_out_max); # close loop EPFL temperature liquid (with maximum heat load)
  
################################
# Variables
################################
var c_factor1{Time} 		>= 0.001; # CarnotFactor, defined as Q/P * ((Tcond/(Tcond-Tevap)))
var c_factor2{Time} 		>= 0.001; # CarnotFactor, calculated as a function of external temperature:  c_factor= -a * T_ext[t]**2 - b *T_ext[t] + c ; 
var a 						>= 0.0000000001; #factor for fitting function for carnot factor 
var b 						>= 0.0000000001;#factor for fitting function for carnot factor 
var c 						>= 0.0000000001; #factor for fitting function for carnot factor 
var se 						>= 0.0000001; # squared error, to be minimized (c_factor1- c_factor2)^2 
var comp1_cost 				>= 0.001 ; 
var comp2_cost 				>= 0.001 ; 
var comp1_cost_2			>= 0.001 ; 
var comp2_cost_2 			>= 0.001 ; 



var Cond_cost				>= 0.001 ; #cost of condenser hex 
var Cond_area				>= 0.001 ; #area of condenser hex 
var Cond_cost_2				>= 0.001 ; #cost of condenser hex 
var Cond_area_2				>= 0.001 ; #area of condenser hex 
var Evap_cost				>= 0.001 ; #cost of evaporator hex 
var Evap_area				>= 0.001 ; #area of evaporator hex 
var Evap_cost_2				>= 0.001 ; #cost of evaporator hex 
var Evap_area_2				>= 0.001 ; #area of evaporator hex 
var DTlnEvap				>= 0.001 ; #logarithmic mean temperature difference of evaporator hex
var DTlnEvap_2				>= 0.001 ; #logarithmic mean temperature difference of evaporator hex
var DTlnCond				>= 0.001 ; #logarithmic mean temperature difference of condenser hex
var DTlnCond_2				>= 0.001 ; #logarithmic mean temperature difference of condenser hex
var TlnCond{Time}           >= 0.001; #logaritmic mean temperature of  temperature loop epfl

################################
# Constraints
################################

#caculates the carnot factor for all time steps of high pressure stage
#################
# !!!!! Fill here: #TlnCond  is the condensation temperature, what is the evaporation temperature of this stage? 
#################
subject to CarnotFactor1{t in Time}:  #caculates the carnot factor for all time steps, avoid dividing by 0 in summer! 
	W_hp[t] >0 ==> c_factor1[t] = Q_cond[t]/ W_hp[t] / (TlnCond[t]+273.15)*(TlnCond[t]-T_evap[t]) else c_factor1[t] =0.001 ;
	
subject to CarnotFactor2{t in Time}:  #caculates the carnot factor for all time steps with fitting function (2nd degree polynomial)
	W_hp[t] > 0 ==> c_factor2[t] = a * (T_ext[t])**2 - b *T_ext[t] + c  else c_factor2[t] =0.001;
	#a * (T_ext[t])**2 - b *T_ext[t] + c  else c_factor2[t] =0.001;
#a+b*log((c*T_ext[t]))
#################
# !!!!! Fill here: temperature difference over condenser
#################
# Condenser
subject to TlnCond_constraint{t in Time}: #calculates the Log mean temperatrure of the epfl  temperature loop 
	(TlnCond[t]+273) * log((T_EPFL_in[t]+273)/(T_EPFL_out[t]+273)) = (T_EPFL_in[t]-T_EPFL_out[t]);
	
subject to DTlnCond_constraint: #calculated the DTLN of the condenser heat exchanger EPFL  temperature loop - heat pump for the expreme period, you can neglect the sensible heat transfer
	DTlnCond * log((T_cond_max-T_EPFL_in_max)/(T_cond_max-T_EPFL_out_max)) = (T_cond_max-T_EPFL_in_max)-(T_cond_max-T_EPFL_out_max);

subject to DTlnCond_constraint_2: #calculated the DTLN of the condenser heat exchanger EPFL  temperature loop - heat pump for the expreme period, you can neglect the sensible heat transfer
	DTlnCond_2 * log((T_cond_min-T_EPFL_in_min)/(T_cond_min-T_EPFL_out_min)) = (T_cond_min-T_EPFL_in_min)-(T_cond_min-T_EPFL_out_min);


#################
# !!!!! Fill here: Condenser area based on maximum condistions, Q=U*A*LMTD
#################
subject to Condenser_area: #Area of condenser HEX, calclated for extreme period 
	Cond_area = Q_cond_max/(U_water_ref*DTlnCond);	

subject to Condenser_cost:
 	Cond_cost = 10^(k1_HEX + k2_HEX*log10(Cond_area) + k3_HEX*(log10(Cond_area))**2) * index /ref_index * f_BM_HEX* (i*(1+i)^n)/((1+i)^n - 1);

subject to Condenser_area_2: #Area of condenser HEX, calclated for extreme period 
	Cond_area_2 = Q_cond_min/(U_water_ref*DTlnCond_2);	

subject to Condenser_cost_2:
 	Cond_cost_2 = 10^(k1_HEX + k2_HEX*log10(Cond_area_2) + k3_HEX*(log10(Cond_area_2))**2) * index /ref_index * f_BM_HEX* (i*(1+i)^n)/((1+i)^n - 1);  	



# Evaporator
subject to DTlnEvap_constraint: #calculated the DTLN of the condenser heat exchanger EPFL  temperature loop - heat pump for the expreme period, you can neglect the sensible heat transfer
	DTlnEvap * log(-(T_evap_max-T_lake_in)/(T_evap_max-T_lake_out)) = -(T_evap_max-T_lake_in)-(T_evap_max-T_lake_out);

subject to DTlnEvap_constraint_2: #calculated the DTLN of the condenser heat exchanger EPFL  temperature loop - heat pump for the expreme period, you can neglect the sensible heat transfer
	 T_evap_min>= T_lake_out ==> DTlnEvap_2 * log(-(T_evap_min-T_lake_in)/(T_evap_min-T_lake_out)) = -(T_evap_min-T_lake_in)-(T_evap_min-T_lake_out) else DTlnEvap_2 * log((T_lake_in-T_evap_min)/(T_lake_out-T_evap_min)) = (T_lake_in-T_evap_min)-(T_lake_out-T_evap_min);
#################
# !!!!! Fill here: Evaporator area based on maximum condistions, Q=U*A*LMTD
#################
subject to Evaporator_area: #Area of evap HEX, calclated for extreme period 
	Evap_area =Q_evap_max/(U_water_ref*DTlnEvap);

 subject to Evaporator_cost:
 	Evap_cost = 10^(k1_HEX + k2_HEX*log10(Evap_area) + k3_HEX*(log10(Evap_area))**2) * index /ref_index * f_BM_HEX*(i*(1+i)^n)/((1+i)^n - 1);

subject to Evaporator_area_2: #Area of evap HEX, calclated for extreme period 
	Evap_area_2 =Q_evap_min/(U_water_ref*DTlnEvap_2);

 subject to Evaporator_cost_2:
 	Evap_cost_2 = 10^(k1_HEX + k2_HEX*log10(Evap_area_2) + k3_HEX*(log10(Evap_area_2))**2) * index /ref_index * f_BM_HEX*(i*(1+i)^n)/((1+i)^n - 1);  	



# Compressor
subject to Comp1cost: 
#calculates the cost for comp1 for extreme period 
 	comp1_cost = 10^(k1 + k2*log10(W_comp1_max) + k3*(log10(W_comp1_max))^2) * index /ref_index * f_BM * (i*(1+i)^n)/((1+i)^n - 1) ;

 subject to Comp2cost: 
#calculates the cost for comp1 for extreme period 
 	comp2_cost = 10^(k1 + k2*log10(W_comp2_max) + k3*(log10(W_comp2_max))^2) * index /ref_index * f_BM * (i*(1+i)^n)/((1+i)^n - 1) ;

# Compressor
subject to Comp1cost_2: 
#calculates the cost for comp1 for extreme period 
 	comp1_cost_2 = 10^(k1 + k2*log10(W_comp1_min) + k3*(log10(W_comp1_min))^2) * index /ref_index * f_BM * (i*(1+i)^n)/((1+i)^n - 1) ;

 subject to Comp2cost_2: 
#calculates the cost for comp1 for extreme period 
 	comp2_cost_2 = 10^(k1 + k2*log10(W_comp2_min) + k3*(log10(W_comp2_min))^2) * index /ref_index * f_BM * (i*(1+i)^n)/((1+i)^n - 1) ;



# Error
 subject to Error: #calculates the mean square error that needs to be minimized 
	se =  sum{t in Time}((c_factor1[t] - c_factor2[t])**2);




################################
minimize obj : se; 