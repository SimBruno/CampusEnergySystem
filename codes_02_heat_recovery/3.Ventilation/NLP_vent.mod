################################
# DTmin optimisation
################################
# Sets & Parameters
reset;
set Time default {};        				#your time set from the MILP 

set Buildings default {};
for {n in {1..24}}{let Buildings:= Buildings union {'Building'&n};}

set MediumTempBuildings default {};								# set of buildings heated by medium temperature loop
set med_T_id = {1, 2, 3, 4, 5 ,6 ,7, 8, 9, 10};
for {n in {med_T_id}}{let MediumTempBuildings:= MediumTempBuildings union {'Building'&n};}

set LowTempBuildings default {};								# set of buildings heated by low temperature loop
set low_T_id = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
for {n in {low_T_id}}{let LowTempBuildings:= LowTempBuildings union {'Building'&n};}


param Text{t in Time};  		  #external temperature - Part 1
param top{Time}; 				  #your operating time from the MILP part

param Tint 				:= 21;    #internal set point temperature [degC]
param mair 				:= 2.5;   #m3/m2/h
param mair_out 			:= 2; 	  #m3/m2/h
param Cpair 			:= 1.152; #kJ/m3K
param Uvent 			:= 0.025; #air-air HEX


param EPFLMediumT 		:= 65;    #[degC]
param EPFLMediumOut 	:= 30;    #[degC]

param CarnotEff 		:= 0.55;  #assumption: carnot efficiency of heating heat pumps
param Cel 				:= 0.20;  #[CHF/kWh] operating cost for buying electricity from the grid

param THPhighin 		:= 7; 	  #[degC] temperature of water coming from lake into the evaporator of the HP
param THPhighout 		:= 3; 	  #[degC] temperature of water coming from lake into the evaporator of the HP
param Cpwater			:= 4.18;  #[kJ/kgC]
#param TLMEvapHP 	:= (THPhighin - THPhighout)/log((THPhighin+273.15)/(THPhighout+273.15)); #[K] logarithmic mean temperature in the evaporator of the heating HP (not using pre-heated lake water

param i 				:= 0.06 ; #interest rate
param n 				:= 20;    #[y] life-time
param FBMHE 			:= 4.74;  #bare module factor of the heat exchanger
param INew 				:= 605.7; #chemical engineering plant cost index (2015)
param IRef 				:= 394.1; #chemical engineering plant cost index (2000)
param aHE 				:= 1200;  #HE cost parameter
param bHE 				:= 0.6;   #HE cost parameter

################################
# Variables

var Text_new{Time} 	;   	  #[degC]
var Trelease{Time}	>= 0; 	  #[degC]
var Qheating{Time} 	>= 0; 	  #your heat demand from the MILP part, is now a variable.

var E{Time} 		>= 0; 	  # [kW] electricity consumed by the heat pump (using pre-heated lake water)
var TLMCond 	 	>= 0.001; #[K] logarithmic mean temperature in the condensor of the heating HP (using pre-heated lake water)
var TLMEvap 		>= 0.001; #[K] logarithmic mean temperature in the evaporator of the heating HP (using pre-heated lake water)
var Qevap{Time} 	>= 0.001; #[kW] heat extracted in the evaporator of the heating HP (using pre-heated lake water)
var Qcond{Time} 	>= 0.001; #[kW] heat delivered in the condensor of the heating HP (using pre-heated lake water)
var COP{Time} 		>= 0.001; #coefficient of performance of the heating HP (using pre-heated lake water)

var OPEX 			>= 0.001; #[CHF/year] operating cost
var CAPEX 			>= 0.001; #[CHF/year] annualized investment cost
var TC 				>= 0.001; #[CHF/year] total cost


var Heat_Vent{Time} >= 0;     #[kW]
var DTLNVent{Time} 	>= 0.001; #[degC]
var Area_Vent 		>= 0.001; #[m2]
var DTminVent 		>= 2; 	  #[degC]

var Flow{Time} 		>= 0; 	  #lake water entering free coling HEX
var MassEPFL{Time} 	>= 0; 	  #MCp of EPFL heating system [KJ/(s degC)]

var Uenv{Buildings} >= 0; 	  #overall heat transfer coefficient of the building envelope 

#### Building dependent parameters

param irradiation{Time};				 # solar irradiation [kW/m2] at each time step									
param specElec{Buildings} default 0.04;  #electricity consumption of buildings [kW/m2]{Buildings}
param FloorArea{Buildings} default 1000; #area [m2]{Buildings}
param k_th{Buildings} default 0.006;	 # thermal losses and ventilation coefficient in (kW/m2/K)
param k_sun{Buildings} default 0.10;	 # solar radiation coefficient [−]{Buildings}
param share_q_e default 0.8; 			 # share of internal gains from electricity [-]{Buildings}
param specQ_people{Buildings} default 0.00226; # specific average internal gains from people [kW/m2]{Buildings}

################################
# Constraints
################################

## VENTILATION

subject to Uenvbuilding{b in MediumTempBuildings}: 
	Uenv[b]+mair*Cpair/3600=k_th[b]; # Uenv calculation for each building based on k_th and mass of air used

subject to VariableHeatdemand {t in Time} : 
	Text[t]<16 ==> Qheating[t]=sum{b in MediumTempBuildings} max(FloorArea[b]*(Uenv[b]*(Tint-Text[t])+(mair/3600)*Cpair*(Tint-Text_new[t])-k_sun[b]*irradiation[t]-specQ_people[b]-share_q_e*specElec[b]),0) else Qheating[t]=0; #Heat demand calculated as the sum of all buildings -> medium temperature sum{b in MediumTempBuildings}
	

subject to Heat_Vent1 {t in Time}: 
	Heat_Vent[t]=sum{b in MediumTempBuildings} ((mair/3600)*FloorArea[b]*Cpair*(Text_new[t]-Text[t])); #HEX heat load from one side;
	
subject to Heat_Vent2 {t in Time}: 
	Heat_Vent[t]=sum{b in MediumTempBuildings} ((mair_out/3600)*FloorArea[b]*Cpair*(Tint-Trelease[t])); #HEX heat load from the other side;
	
subject to DTLNVent1 {t in Time}: 
	DTLNVent[t] * log((Tint-Text_new[t])/(Trelease[t]-Text[t])) = ((Tint-Text_new[t])-(Trelease[t]-Text[t])); #DTLN ventilation -> pay attention to this value: why is it special?

#subject to dTLNVent_rule{t in Time}: DTLNVent[t]<=((Text_new[t]-Tint+0.001)+273+(Text[t]-Trelease[t]+0.001)+273)/2; # One of inequalities for Condenser


#subject to dTLNVent_rule2{t in Time}: DTLNVent[t]>=((Text_new[t]-Tint+0.001)*(Text[t]-Trelease[t]+0.001))**0.5; # The other inequality for Condenser
	

subject to Area_Vent1 {t in Time}: 
	Area_Vent*Uvent*DTLNVent[t]>=Heat_Vent[t]; #Area of ventilation HEX
	
subject to DTminVent1 {t in Time}: 
	Trelease[t]>=DTminVent+Text[t]; #DTmin needed on one side of HEX
	
subject to DTminVent2 {t in Time}: 
	Tint>=DTminVent+Text_new[t]; #DTmin needed on the other side of HEX 
	
subject to ventilation_trivial {t in Time}: #relation between Text_new and Text (initialization purposes)
	Text_new[t] >= Text[t]+0.01;

## MASS BALANCE

subject to Flows{t in Time}: 
	MassEPFL[t]*(EPFLMediumT-EPFLMediumOut)=Qheating[t]; #MCp of EPFL heating fluid calculation.

## MEETING HEATING DEMAND, ELECTRICAL CONSUMPTION

subject to QEvaporator{t in Time}: 
	Qevap[t]=Flow[t]*Cpwater*(THPhighin-THPhighout); #water side of evaporator that takes flow from lake (Reference case)

subject to QCondensator{t in Time}: 
	Qcond[t]=MassEPFL[t]*(EPFLMediumT-EPFLMediumOut); #EPFL side of condenser delivering heat to EFPL (Reference case)
		
subject to Electricity1{t in Time}: 
	E[t]+Qevap[t]=Qcond[t]; #the electricity consumed in the HP can be computed using the heat delivered and the heat extracted (Reference case)

subject to Electricity{t in Time}: 
	E[t]*COP[t]=Qcond[t]; #the electricity consumed in the HP can be computed using the heat delivered and the COP (Reference case)
	
subject to COPerformance{t in Time}: 
	COP[t]*(TLMCond-TLMEvap)=CarnotEff*(TLMCond); #the COP can be computed using the carnot efficiency and the logarithmic mean temperatures in the condensor and in the evaporator (Reference case)
	
subject to dTLMCondensor{t in Time}:
	TLMCond=(EPFLMediumOut-EPFLMediumT)/log((EPFLMediumOut+273)/(EPFLMediumT+273)); #the logarithmic mean temperature on the condenser, using inlet and outlet temperatures. Note: should be in K (Reference case)

subject to dTLMEvaporatorHP{t in Time}:
	TLMEvap=(THPhighin-THPhighout)/log((THPhighin+273)/(THPhighout+273)); #the logarithmic mean temperature can be computed using the inlet and outlet temperatures, Note: should be in K (Reference case)
	
## MEETING HEATING DEMAND, ELECTRICAL CONSUMPTION

subject to QEPFLausanne{t in Time}:
	Qheating[t]=Qcond[t]; #the heat demand of EPFL should be supplied by the the HP.
	
subject to OPEXcost:
	OPEX = sum{t in Time}(Cel*top[t]*E[t]); #the operating cost can be computed using the electricity consumed in the HP.
	
subject to CAPEXcost: CAPEX*IRef=((i*(1+i)^n)/((1+i)^n-1))*(INew)*FBMHE*aHE*Area_Vent^bHE; #the investment cost can be computed using the area of the ventilation heat exchanegr
	

subject to TCost: 
	TC=OPEX+CAPEX; #the total cost can be computed using the operating and investment cost
	
################################
minimize obj : TC;
