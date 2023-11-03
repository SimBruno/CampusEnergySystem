################################
# DTmin optimisation
################################
# Sets & Parameters
reset;
set Time default {};        				#your time set from the MILP 

set Buildings default {};
for {n in {1..24}}{let Buildings:= Buildings union {'Building'&n};}

set MediumTempBuildings default {};								# set of buildings heated by medium temperature loop
set med_T_id = {1, 2, 3, 4, 5 ,6 ,7, 8, 9};
for {n in {med_T_id}}{let MediumTempBuildings:= MediumTempBuildings union {'Building'&n};}

set LowTempBuildings default {};								# set of buildings heated by low temperature loop
set low_T_id = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
for {n in {low_T_id}}{let LowTempBuildings:= LowTempBuildings union {'Building'&n};}


param Text{t in Time};  #external temperature - Part 1
param top{Time}; 		#your operating time from the MILP part

param Tint 				:= 21; # internal set point temperature [C]
param mair 				:= 2.5; # m3/m2/h
param Cpair 			:= 1.152; # kJ/m3K
param Uvent 			:= 0.025; # air-air HEX


param EPFLMediumT 		:= 65+273.15; #[degC]
param EPFLMediumOut 	:= 30+273.15; #[degC]

param CarnotEff 		:= 0.55; #assumption: carnot efficiency of heating heat pumps
param Cel 				:= 0.20; #[CHF/kWh] operating cost for buying electricity from the grid

param THPhighin 		:= 7+273.15; #[K] temperature of water coming from lake into the evaporator of the HP
param THPhighout 		:= 3+273.15; #[K]] temperature of water coming from lake into the evaporator of the HP
param Cpwater			:= 4.18; #[kJ/kgC]
#param TLMEvapHP 	:= (THPhighin - THPhighout)/log((THPhighin+273.15)/(THPhighout+273.15)); #[K] logarithmic mean temperature in the evaporator of the heating HP (not using pre-heated lake water

param i 				:= 0.06 ; #interest rate
param n 				:= 20; #[y] life-time
param FBMHE 			:= 4.74; #bare module factor of the heat exchanger
param INew 				:= 605.7; #chemical engineering plant cost index (2015)
param IRef 				:= 394.1; #chemical engineering plant cost index (2000)
param aHE 				:= 1200; #HE cost parameter
param bHE 				:= 0.6; #HE cost parameter

####OUR parameters

param TLMCond := (EPFLMediumOut-EPFLMediumT)/(log(EPFLMediumOut/EPFLMediumT));
param TLMEvapHP = (THPhighin-THPhighout)/(log((THPhighin)/(THPhighout))); #Q:Counterflow for Evap, even if the scheme shows it differently
param COP := CarnotEff/(1-TLMEvapHP/TLMCond);




################################
# Variables

var Text_new{Time} 	>= 0; #[degC]
var Trelease{Time}	>= 0; #[degC]
var Qheating{Time} 	>= 0; #your heat demand from the MILP part, is now a variable.

var E{Time} 		>= 0; # [kW] electricity consumed by the heat pump (using pre-heated lake water)
#var TLMCond 	 	>= 273.15+30.001; #[K] logarithmic mean temperature in the condensor of the heating HP (using pre-heated lake water)
#var TLMEvap 		>= 273.15+0.001; # K] logarithmic mean temperature in the evaporator of the heating HP (using pre-heated lake water)
var Qevap{Time} 	>= 0.001; #[kW] heat extracted in the evaporator of the heating HP (using pre-heated lake water)
var Qcond{Time} 	>= 0.001; #[kW] heat delivered in the condensor of the heating HP (using pre-heated lake water)
#var COP{Time} 		>= 0.001; #coefficient of performance of the heating HP (using pre-heated lake water)

var OPEX 			>= 0.001; #[CHF/year] operating cost
var CAPEX 			>= 0.001; #[CHF/year] annualized investment cost
var TC 				>= 0.001; #[CHF/year] total cost


var Heat_Vent{Time} >= 0; #[kW]
var DTLNVent{Time} 	>= 0.001; #[degC]
var Area_Vent 		>= 0.001; #[m2]
var DTminVent 		>= 2; #[degC]

var Flow{Time} 		>= 0; #lake water entering free coling HEX
var MassEPFL{Time} 	>= 0; # MCp of EPFL heating system [KJ/(s degC)]

var Uenv{Buildings} >= 0; # overall heat transfer coefficient of the building envelope 

#### Building dependent parameters

param irradiation{Time};# solar irradiation [kW/m2] at each time step									
param specElec{Buildings} default 0.04;
param FloorArea{Buildings} default 1000; #area [m2]
param k_th{Buildings} default 0.006; # thermal losses and ventilation coefficient in (kW/m2/K)
param k_sun{Buildings} default 0.10;# solar radiation coefficient [−]
param share_q_e default 0.8; # share of internal gains from electricity [-]
param specQ_people{Buildings} default 0.007;# specific average internal gains from people [kW/m2]

################################
# Constraints
################################

## VENTILATION

subject to Uenvbuilding{b in MediumTempBuildings}: # Uenv calculation for each building based on k_th and mass of air used
	Uenv[b] = k_th[b]-mair*Cpair/3600;#k_th = total losses, MCp = Conduction losses, Unev = convection losses
	

subject to VariableHeatdemand {t in Time} : #Heat demand calculated as the sum of all buildings -> medium temperature
	Qheating[t] = sum{b in MediumTempBuildings}max{FloorArea[b]*(Uenv[b]*(Tint-Text[t])+mair*Cpair*(Tint-Text_new[t])-k_sun*irradiation[t]-specQ_people[b])-share_q_e*specElec[b]};

subject to Heat_Vent1 {t in Time}: #HEX heat load from one side;
	Heat_Vent[t] = (Text_new[t]-Text[t])*mair*Cpair/3600; #>0

subject to Heat_Vent2 {t in Time}: #HEX heat load from the other side;
	Heat_Vent[t] = 0.8*(Tint-Trelease[t])*mair*Cpair/3600; #>0

subject to DTLNVent1 {t in Time}: #DTLN ventilation -> pay attention to this value: why is it special?
	DTLNVent[t] = ((Tint-Text_new[t])-(Trelease[t]-Text[t]))/log((Tint-Text_new[t])/(Trelease[t]-Text[t]));

subject to Area_Vent1 {t in Time}: #Area of ventilation HEX
	Area_Vent[t] = Heat_Vent[t]/(Uvent*DTLNVent[t]);

subject to DTminVent1 {t in Time}: #DTmin needed on one side of HEX
	DTminVent[t] = (Text_new[t]-Text[t])/log(Text_new[t]/Text[t]);

subject to DTminVent2 {t in Time}: #DTmin needed on the other side of HEX 
	DTminVent[t] = (Tint-Trelease[t])/log(Tint/Trelease[t]);

subject to ventilation_trivial {t in Time}: #relation between Text_new and Text (initialization purposes)
	Text_new[t] >= Text[t]+0.01;
## MASS BALANCE

subject to Flows{t in Time}: #MCp of EPFL heating fluid calculation.
	MassEPFL[t] =Qheating[t]/(EPFLMediumT-EPFLMediumT);	

## MEETING HEATING DEMAND, ELECTRICAL CONSUMPTION

subject to QEvaporator{t in Time}: #water side of evaporator that takes flow from lake (Reference case)
	Qevap[t] = Flow[t] * (THPhighin-THPhighout) #2 eq and 3 Unkowns
    ; ##>0

subject to QCondensator{t in Time}: #EPFL side of condenser delivering heat to EFPL (Reference case)
	Qcond[t] = MassEPFL[t] * (EPFLMediumT-EPFLMediumOut) #3eq and 4 Unkowns
    ; #> 0

subject to Electricity1{t in Time}: #the electricity consumed in the HP can be computed using the heat delivered and the heat extracted (Reference case)
	E[t] = -Qevap[t] + Qcond[t]; 

subject to Electricity{t in Time}: #the electricity consumed in the HP can be computed using the heat delivered and the COP (Reference case)
	E[t] = Qcond[t] / COP;

#subject to COPerformance{t in Time}: #the COP can be computed using the carnot efficiency and the logarithmic mean temperatures in the condensor and in the evaporator (Reference case)
#	1/COP = 1 /CarnotEff * (1-TLMEvapHP/TLMCond);

#subject to dTLMCondensor{t in Time}: #the logarithmic mean temperature on the condenser, using inlet and outlet temperatures. Note: should be in K (Reference case)
#	TLMCond = ((TCondensorin-EPFLMediumOut)-(TCondenserout[t]-EPFLMediumT))/(log((TCondensorin-EPFLMediumOut)/(TCondenserout-EPFLMediumT)));

#subject to dTLMEvaporatorHP{t in Time}: #the logarithmic mean temperature can be computed using the inlet and outlet temperatures, Note: should be in K (Reference case)
#	TLMEvapHP = ((TEvapin-THPhighout)-(TEvapout-THPhighin))/(log((TEvapin-THPhighout)/(TEvapout-THPhighin))); 

## MEETING HEATING DEMAND, ELECTRICAL CONSUMPTION

#Q: WHAT IS THIS, pls?
#subject to QEPFLausanne{t in Time}: #the heat demand of EPFL should be supplied by the the HP.
	

subject to OPEXcost: #the operating cost can be computed using the electricity consumed in the HP.
	OPEX = sum{t in Time}{Qheating[t]*Cel*top[t]};

#Q: how does CAPEX depend on Vent_area
subject to CAPEXcost: #the investment cost can be computed using the area of the ventilation heat exchanegr
	CAPEX = sum{t in Time}max{(FBMHE*(i*(i+1)^n)/((i+1)^n-1)*(INew/IRef)*aHE*AHEDC^bHE)*Area_Vent[t]};
	
subject to TCost: #the total cost can be computed using the operating and investment cost
	TC = CAPEX+OPEX;

################################
minimize obj : TC;
