/*---------------------------------------------------------------------------------------------------------------------------------------
Course: Modelling and optimisation of energy systems course spring semester 2019
EPFL Campus energ system optimization
IPESE, EPFL
---------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------
This is a model which will be the base of the project. It has the definition of the generic sets, parameters and variables. It also
includes equations that apply to the whole system. Depending on the modifications and additions to the model, the constraints are 
subject to change.
---------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------
Generic Sets
---------------------------------------------------------------------------------------------------------------------------------------*/
set Buildings default {};
for {n in {1..24}}{let Buildings:= Buildings union {'Building'&n};}

set MediumTempBuildings default {};								# set of buildings heated by medium temperature loop
set med_T_id = {1, 2, 3, 4, 5 ,6 ,7, 8, 9};
for {n in {med_T_id}}{let MediumTempBuildings:= MediumTempBuildings union {'Building'&n};}

set LowTempBuildings default {};								# set of buildings heated by low temperature loop
set low_T_id = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
for {n in {low_T_id}}{let LowTempBuildings:= LowTempBuildings union {'Building'&n};}


set Time default {}; 											# time segments of the problem 
set Technologies default {};									# technologies to provide heating cooling and elec
set Grids default {};											# grid units to buy resources (fuel, electricity etc.)
set Utilities := Technologies union Grids;						# combination of technologies and grids
set Layers default {};											# resources to provide fuel, elec, etc.
set HeatingLevel default {};									# low and medium temlerature levels
set UtilityType default {};										# type of the utility: heating, cooling, elec	
set UtilitiesOfType{UtilityType} default {};					# utilities assigned to their respective utility type
set UtilitiesOfLayer{Layers} default {};

/*---------------------------------------------------------------------------------------------------------------------------------------
Generic Parameters
---------------------------------------------------------------------------------------------------------------------------------------*/
param dTmin default 5;											# minimum temperature difference in the heat exchangers
param top{Time};												# operating time [h] of each time step
param Theating{HeatingLevel}	;								# temperatue [C] of the heating levels
param irradiation{Time};										# solar irradiation [kW/m2] at each time step
param roofArea default 15500;									# available roof area for PV installation
param refSize default 1000;									    # reference size of the utilities [kW]
param Text{t in Time};  
param Tint default 21;											# internal set point temperature [C]
param specElec{Buildings} default 0.04;
param a_ex{Utilities} default 0;
param b_ex{Utilities} default 0;
param c_ex{Utilities} default 1;

param Max_Emissions default 1e20;
param Max_Totalcost default 1e20;

# Heat pumps data

param EPFLMediumT 	:= 338; #[degK] - desired temperature high temperature loop
param EPFLLowT		:= 323;
param EPFLMediumOut := 303; # temperature of return low temperature loop [degK]
param THPhighin 	:= 280; #[deg K] temperature of water coming from lake into the evaporator of the HP
param THPhighout 	:= 276; #[deg K] temperature of water coming from lake into the evaporator of the HP
param TLMCondMedium := (EPFLMediumOut-EPFLMediumT)/(log(EPFLMediumOut/EPFLMediumT)); #Assume cste
param TLMCondLow := (EPFLMediumOut-EPFLLowT)/(log(EPFLMediumOut/EPFLLowT)); #Assume cste
param TLMEvapHP := (THPhighin-THPhighout)/(log((THPhighin)/(THPhighout))); 

/*---------------------------------------------------------------------------------------------------------------------------------------
Calculation of heating demand
---------------------------------------------------------------------------------------------------------------------------------------*/
param FloorArea{Buildings} default 1000;
param k_th{Buildings} default 0.006;								# thermal losses and ventilation coefficient in (kW/m2/K)
param k_sun{Buildings} default 0.10;								# solar radiation coefficient [−]
param share_q_e default 0.8; 										# share of internal gains from electricity [-]
param specQ_people{Buildings} default 0.007;						# specific average internal gains from people [kW/m2]
param Qheating{b in Buildings, t in Time} :=
		if Text[t] < 16  then 
			max(FloorArea[b]*(k_th[b]*(Tint-Text[t]) - k_sun[b]*irradiation[t]-specQ_people[b] - share_q_e*specElec[b]),0)
		else
			0
;
param Qheatingdemand{h in HeatingLevel, t in Time} :=
	if h == 'MediumT' then
		sum{b in MediumTempBuildings} Qheating[b,t]
	else
		sum{b in LowTempBuildings} Qheating[b,t]
	;

/*---------------------------------------------------------------------------------------------------------------------------------------
Calculation of electricity demand
---------------------------------------------------------------------------------------------------------------------------------------*/

param Ebuilding{b in Buildings, t in Time} :=
	specElec[b] * FloorArea[b];
param Edemand{t in Time} :=
	sum{b in Buildings} Ebuilding[b,t];

/*---------------------------------------------------------------------------------------------------------------------------------------
Utility paremeters
---------------------------------------------------------------------------------------------------------------------------------------*/
# minimum temperature of the heating technologies [C]
param Tminheating{UtilitiesOfType['Heating']} default 90;

# reference flow of the heating and cooling [kW]
param Qheatingsupply{UtilitiesOfType['Heating']} default 1000;

# reference flow of the resources (elec, natgas etc) [kW] [m3/s] [kg/s]
param Flowin{l in Layers,u in UtilitiesOfLayer[l]} default 0;
param Flowout{l in Layers,u in UtilitiesOfLayer[l]} default 0;

# minimum and maximum scaling factors of the utilities
param Fmin{Utilities} default 0.001;
param Fmax{Utilities} default 1000;

/*---------------------------------------------------------------------------------------------------------------------------------------
Utility variables
---------------------------------------------------------------------------------------------------------------------------------------*/
var use{Utilities} binary;										# binary variable to decide if a utility is used or not
var use_t{Utilities, Time} binary;								# binary variable to decide if a utility is used or not at time t
var mult{Utilities}>=0;											# continuous variable to decide the size of the utility
var mult_t{Utilities, Time}>=0;									# continuous variable to decide the size of the utility at time t

/*---------------------------------------------------------------------------------------------------------------------------------------
Resource variables
---------------------------------------------------------------------------------------------------------------------------------------*/
var FlowInUnit{Layers,Utilities,Time} >= 0;						# continuous variables to decide the size of the resource demand
var FlowOutUnit{Layers,Utilities,Time} >= 0;

/*---------------------------------------------------------------------------------------------------------------------------------------
Utility sizing constraints
---------------------------------------------------------------------------------------------------------------------------------------*/
subject to size_cstr1{u in Utilities, t in Time}: 				# coupling the binary variable with the continuous variable 1
	Fmin[u]*use_t[u,t] <= mult_t[u, t];

subject to size_cstr2{u in Utilities, t in Time}:				# coupling the binary variable with the continuous variable 2
	mult_t[u, t] <= Fmax[u]*use_t[u, t];

subject to size_cstr3{u in Utilities, t in Time}: 				# size in each time should be less than the nominal size
	mult_t[u, t] <= mult[u];

subject to size_cstr4{u in Utilities}:							# coupling the binary variable with the continuous variable 2
	mult[u] <= Fmax[u]*use[u];

subject to size_cstr5{u in Utilities}: 							# coupling the binary variable with the continuous variable 1
	Fmin[u]*use[u] <= mult[u];

subject to size_cstr6{u in Utilities}: 				# limitzing the binary variable use
	use[u] <= sum{t in Time} use_t[u,t];	

subject to Not_all_HP:
 	use['R1270_LT']+use['R1270_MT']+use['R290_LT']+use['R290_MT']<=1;

/*---------------------------------------------------------------------------------------------------------------------------------------
Heating balance constraints: demand = supply
---------------------------------------------------------------------------------------------------------------------------------------*/
# heating balance: demand = supply
var mult_heating_t{UtilitiesOfType['Heating'], Time, HeatingLevel} >= 0;

subject to LT_balance{t in Time}:
	Qheatingdemand['LowT',t] = sum{u in UtilitiesOfType['Heating']: Tminheating[u] >= Theating['LowT'] + dTmin} (Qheatingsupply[u] * mult_heating_t[u,t,'LowT']);

subject to MT_balance{t in Time}:
	Qheatingdemand['MediumT',t] = sum{u in UtilitiesOfType['Heating']: Tminheating[u] >= Theating['MediumT'] + dTmin} (Qheatingsupply[u] * mult_heating_t[u,t,'MediumT']);

subject to heaitng_mult_cstr{u in UtilitiesOfType['Heating'], t in Time}:
	mult_t[u,t] = sum{h in HeatingLevel} mult_heating_t[u,t,h];

	

# the following two constraints are to ensure that one utility will not be usd if the supply temperature is less than needed.
subject to zero_constraint1{t in Time}:
	sum{u in UtilitiesOfType['Heating']: Tminheating[u] <= Theating['MediumT'] + dTmin} mult_heating_t[u,t,'MediumT'] = 0;

subject to zero_constraint2{t in Time}:
	sum{u in UtilitiesOfType['Heating']: Tminheating[u] <= Theating['LowT'] + dTmin} mult_heating_t[u,t,'LowT'] = 0;


/*---------------------------------------------------------------------------------------------------------------------------------------
Resource balance constraints (except for electricity): flowin = flowout
---------------------------------------------------------------------------------------------------------------------------------------*/



subject to inflow_cstr {l in Layers, u in UtilitiesOfLayer[l], t in Time}:
	if u!= 'R1270_LT' and u!= 'R1270_MT' and u!= 'R290_LT' and u!= 'R290_MT' and u!='STC' then
		FlowInUnit[l, u, t] = mult_t[u,t] * Flowin[l,u];


subject to outflow_cstr {l in Layers, u in UtilitiesOfLayer[l], t in Time}:
	if u!= 'R1270_LT' and u!= 'R1270_MT' and u!= 'R290_LT' and u!= 'R290_MT' and u!='STC' then
		FlowOutUnit[l, u, t] = mult_t[u,t] * Flowout[l,u]; 

	
subject to balance_cstr {l in Layers, t in Time: l != 'Electricity'}:
	sum{u in UtilitiesOfLayer[l]} FlowInUnit[l,u,t] = sum{u in UtilitiesOfLayer[l]} FlowOutUnit[l,u,t];

/*---------------------------------------------------------------------------------------------------------------------------------------
Electricity balance constraints: building demand + utility cons = utility supply 
---------------------------------------------------------------------------------------------------------------------------------------*/
subject to electricity_balance{t in Time}:
	Edemand[t] + sum{u in UtilitiesOfType['ElectricityCons']} FlowInUnit['Electricity',u,t] = sum{u in UtilitiesOfType['ElectricitySup']} FlowOutUnit['Electricity',u,t];

/*---------------------------------------------------------------------------------------------------------------------------------------
Cost parameters and constraints
---------------------------------------------------------------------------------------------------------------------------------------*/
param c_spec{g in Grids}
default 
if g='NatGasGrid' then 0.0303
else if  g='ElecGridBuy' then 0.0916
else if g='ElecGridSell' then -0.06
else if g='HydrogenGrid' then 0.3731
else 0.001;

# let c_spec{NatGasGrid} default 0.0303;
# let c_spec{ElecGridBuy} default 0.0916;
# let c_spec{ElecGridSell} default -0.06;
# let c_spec{HydrogenGrid} default 0.3731;

param cop2g{g in Grids} = c_spec[g] * refSize;					# mult_t dependent cost of the reosurce [CHF/kWh * refSize]

param cop1t{Technologies} default 0.001;							# fixed cost of the technology [CHF/h]
param cop2t{Technologies} default 0.001;							# variable cost of the technology [CHF/h]

param cop1{u in Utilities} = 									# fixed cost of the utility [CHF/h]
	if (exists{g in Grids} (g = u)) then 
		0.001 
	else 
		cop1t[u]
	;
param cop2{u in Utilities} = 									# variable cost of the utility [CHF/h]
	if (exists{g in Grids} (g = u)) then 
		cop2g[u] 
	else 
		cop2t[u]
	;
param cinv1{t in Technologies} default 0.001;						# fixed investment cost of the utility [CHF/year]
param cinv2{t in Technologies} default 0.001;						# variable investment cost of the utility [CHF/year]

param c_elec default 75.3; #electricity emissions in Switzerland 13/12/2023[gCO2/kWh] --> https://www.horocarbon.ch/mix.php
param c_gas default 228; #natural gas emissions in Switzerland 2018[gCO2/kWh] --> https://www.wwf.ch/sites/default/files/doc-2018-10/2018-06-Factsheet-NaturalGas-Biogas-PtG.pdf
param CO2tax default 120; #CO2 tax in Switzerland 2022 [CHF/tCO2] --> https://www.iea.org/policies/17762-swiss-carbon-tax#

#param eff{Technologies diff {"Cogen","SOFC","HP1stageLT","HP1stageMT"}} default 0.9;		# efficiency of each technology, these values are not definitive ones. 
#param cop{{"HP1stageLT","HP1stageMT"}, Time} default 3;		# efficiency of each technology, these values are not definitive ones. 

# variable and constraint for operating cost calculation [CHF/year]
var OpCost;
subject to oc_cstr:
	OpCost = sum {u in Utilities, t in Time} (cop1[u] * use_t[u,t] + cop2[u] * mult_t[u,t]) * top[t];

# variable and constraint for investment cost calculation [CHF/year]
var InvCost;
subject to ic_cstr:
	InvCost = sum{tc in Technologies} (cinv1[tc] * use[tc] + cinv2[tc] * mult[tc]);

#variable and constraint for emissions calculation [gCO2/year]
var Emissions;
var Totalcost;
subject to em_cstr:
	Emissions = sum{u in Utilities, t in Time} (FlowInUnit['Electricity',u,t] * top[t] * c_elec)
	+ sum{u in Utilities, t in Time} (FlowInUnit['Natgas',u,t] * top[t] * c_gas);

subject to max_emission_cstr:
	Emissions <= Max_Emissions;

subject to Totalcost_value:
	Totalcost=InvCost+OpCost;

subject to max_totalcost_cstr:
	Totalcost<=Max_Totalcost;

/*---------------------------------------------------------------------------------------------------------------------------------------
Objective function
---------------------------------------------------------------------------------------------------------------------------------------*/
#minimize Totalcost:InvCost + OpCost;# + Emissions*CO2tax*10^(-6);

#subject to test_cstr:
#	use['R290_MT']=0;

#subject to test_cstr2:
#	use['Boiler']=0;


# subject to test_cstr3:
# 	use['SOFC']=1;


