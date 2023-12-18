/*---------------------------------------------------------------------------------------------------------------------------------------
Set the efficiency of PV
---------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------
Set the electricity output as a function of irradiation and efficiency 
---------------------------------------------------------------------------------------------------------------------------------------*/
var usedroofAreaPV >= 0;
var usedroofAreaSTC >= 0;

param eff_PV := 0.15;
param eff_STC := 0.836;
param Tlm := 61.22;

subject to pv_elec{t in Time}:
	Flowout['Electricity','PV'] * mult_t['PV',t] = irradiation[t] * usedroofAreaPV * eff_PV;
	Qheatingsupply['STC'] * mult_t['STC',t] = usedroofAreaSTC * (irradiation[t] * eff_STC - 4.16/1000*(Tlm-Text[t]) - 0.073/1000*(Tlm-Text[t])^2);
	cinv1["STC"]:=0.087*347*usedroofAreaSTC; # CHF/year sr: Stadler P.,  Model-based sizing of building energy systems with renewable sources
	cinv2["STC"]:=0.087*126*usedroofAreaSTC; # CHF/kW (1000W/m2 and 1 m2) sr: Stadler P., Model-based sizing of building energy systems with renewable sources

/*---------------------------------------------------------------------------------------------------------------------------------------
roof usage 
---------------------------------------------------------------------------------------------------------------------------------------*/
subject to roof_cstr:
roofArea >= usedroofAreaPV + usedroofAreaSTC;