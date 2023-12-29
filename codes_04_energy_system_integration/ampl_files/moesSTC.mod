

param eff_STC := 0.75;#0.836;
param Tlm := 61.22;


subject to STC_heat_irr{t in Time}:
	irradiation[t] =0 ==> usedroofAreaSTC[t]=0;

subject to STC_heat{t in Time}:
	Qheatingsupply['STC'] * mult_t['STC',t] = usedroofAreaSTC[t] * (irradiation[t] * eff_STC);# - 4.16/1000*(Tlm-Text[t]) - 0.073/1000*(Tlm-Text[t])^2,0);

subject to roof_cstr2{t in Time}:
	max_usedroofAreaSTC>= usedroofAreaSTC[t];

#let cinv1["STC"]:=0.087*347;#*usedroofAreaSTC; # CHF/year sr: Stadler P.,  Model-based sizing of building energy systems with renewable sources
#let cinv2["STC"]:=0.087*126*usedroofAreaSTC/mult['STC']; # CHF/year (1000W/m2 and 1 m2) sr: Stadler P., Model-based sizing of building energy systems with renewable sources
#let cinv2["STC"]:=0.087*126*3.61;