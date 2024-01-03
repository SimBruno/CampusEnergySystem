reset;
################################
# DTmin optimisation
################################
# Sets & Parameters

set Time default {};        #your time set from the MILP part
param Qheating{Time};       #your heat demand from the MILP part, only Medium temperature heat (65 deg C) [kW or kWh]
param top{Time};            #your operating time from the MILP part [hours]

param EPFLMediumT 	:= 338; #[degC] - desired temperature high temperature loop
param EPFLMediumOut := 303; #temperature of return low temperature loop [degC]

param CarnotEff 	:= 0.55;#assumption: carnot efficiency of heating heat pumps
param Cel 			:= 0.20;#[CHF/kWh] operating cost for buying electricity from the grid

param THPhighin 	:= 280; #[deg C] temperature of water coming from lake into the evaporator of the HP
param THPhighout 	:= 276; #[deg C] temperature of water coming from lake into the evaporator of the HP

#Can add these parameters, because we assume constant temperature within the haet pump, to change the heat load one will change the mass flow, constant Temp lead to constant COP
#TLM of cond and evap, not of the hex 

param TLMCond := (EPFLMediumOut-EPFLMediumT)/(log(EPFLMediumOut/EPFLMediumT));
#param TLMCond := (EPFLMediumOut+EPFLMediumT)/2;
param TLMEvapHP = (THPhighin-THPhighout)/(log((THPhighin)/(THPhighout))); #Q:Counterflow for Evap, even if the scheme shows it differently
#param TLMEvapHP = (THPhighin+THPhighout)/2; #Q:Counterflow for Evap, even if the scheme shows it differently
param COP := CarnotEff*TLMCond/(TLMCond-TLMEvapHP);


################################
# Variables

#3 variables were eliminated, because they were taken as fixed

var E{Time} >= 0.001;        #[kW] electricity consumed by the heat pump (using pre-heated lake water)
#var TLMCond{Time} 	>= 0.001; #[K] logarithmic mean temperature in the condensor of the heating HP (using pre-heated lake water)
var Qevap{Time} >= 0.001;    #[kW] heat extracted in the evaporator of the heating HP (using pre-heated lake water)
var Qcond{Time} >= 0.001;    #[kW] heat delivered in the condensor of the heating HP (using pre-heated lake water)

#var COP{Time} 		>= 0.001; #coefficient of performance of the heating HP (using pre-heated lake water)
var OPEX{Time} >= 0.001;     #[CHF/year] operating cost
#var TLMEvapHP{Time} >= 0.001; #[K] logarithmic mean temperature in the evaporator of the heating HP
var Flow{Time} >= 0.001;     #lake water entering free coling HEX [kg/s]
var MassEPFL{Time} >= 0.001; #MCp of EPFL heating system [KJ/(s degC)]

###TESTS: Could be added as vairables

#var TCondensorin{Time} >= 0.001; # If we choose to have TLM that can change, can also choose just massflow to change
#var TCondenserout{Time}>= 0.001;
#var TEvapout{Time}>= 0.001;
#var TEvapout{Time}>= 0.001;

################################
# Constraints
################################
minimize Totopex : 
    sum{t in Time} (OPEX[t]);

## MASS BALANCE
subject to Flows{t in Time}: #MCp of EPFL heating fluid calculation.
   MassEPFL[t] = Qheating[t]/top[t]/(EPFLMediumT-EPFLMediumOut); #1 eq & 1 Unknown [KJ/(s degC)] from enregy balance of EPFLMediumT

## MEETING HEATING DEMAND, ELECTRICAL CONSUMPTION

subject to QEvaporator{t in Time}: #water side of evaporator that takes flow from lake
    Qevap[t] = Flow[t] * (THPhighin-THPhighout); #2 eq and 3 Unkowns

subject to QCondensator{t in Time}: #EPFL side of condenser delivering heat to EFPL
    Qcond[t] = MassEPFL[t] * (EPFLMediumT-EPFLMediumOut); #3eq and 4 Unkowns

 #W = Electrictiy consumed
subject to Electricity1{t in Time}: #the electricity consumed in the HP (using pre-heated lake water) can be computed using the heat delivered and the heat extracted
    E[t] + Qevap[t] = Qcond[t]; #4eq and 5 Unkowns

subject to Electricity{t in Time}: #the electricity consumed in the HP (using pre-heated lake water) can be computed using the heat delivered and the COP
    E[t] = Qcond[t] / COP; #5 eq and 5 Unkowns

# These 3 are taken as parameters and can be eliminated

#subject to COPerformance{t in Time}: #the COP can be computed using the carnot efficiency and the logarithmic mean temperatures in the condensor and in the evaporator
#    1/COP = 1 /CarnotEff * (1-TLMEvapHP/TLMCond);

#subject to dTLMCondensor{t in Time}: #the logarithmic mean temperature on the condenser, using inlet and outlet temperatures. Note: should be in K
 #   TLMCond = ((TCondensorin-EPFLMediumOut)-(TCondenserout[t]-EPFLMediumT))/(log((TCondensorin-EPFLMediumOut)/(TCondenserout-EPFLMediumT)));

#subject to dTLMEvaporatorHP{t in Time}: #the logarithmic mean temperature can be computed using the inlet and outlet temperatures, Note: should be in K
#    TLMEvapHP = ((TEvapin-THPhighout)-(TEvapout-THPhighin))/(log((TEvapin-THPhighout)/(TEvapout-THPhighin))); 


#Do not need this equation, its the same as QCondensator?
#subject to QEPFLausanne{t in Time}: #the heat demand of EPFL should be supplied by the the HP.
#    Qcond[t] = Qheating[t];

subject to OPEXcost{t in Time}: #the operating cost can be computed using the electricity consumed in the HP;
    OPEX[t] = Cel * E[t] * top[t]; #7 eq and 7 unkowns
################################

