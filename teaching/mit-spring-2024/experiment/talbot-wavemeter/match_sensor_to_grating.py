# lines_per_um = input("Grating lines per micron ")
# try:
	# period = 1 / int(lines_per_um)
# except:
    # period = float(input("Period in um "))
    
lines_per_um = [1.2, 0.83, 0.6, 0.3]

for n in lines_per_um:
    period = 1 / n
    # Talbot signal will be maximized when the pixel pitch is an odd multiple of half the grating period
    multiples = [1, 3, 5, 7, 9, 11, 13]
    badmultiples = [2, 4, 6, 8, 10, 12, 14]
    half_period = period / 2
    pitches = [round(half_period * multiple, 3) for multiple in multiples]
    badpitches = [round(half_period * multiple, 3) for multiple in badmultiples]
    print("Lines per um:", n)
    print(f"Line periodicity: {period:.2f}")
    print("Accepted pixel pitches:", pitches)
    print("Unacceptable pixel pitches:", badpitches)
    print("\n")