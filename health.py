
def loader():

	file = open("data.txt","r") 
	contents=file.read()
		
		

		#print(contents)
		#print(contents)
		#creates array with each data set 
	totalList=list()
	for x in range(0, 30000):
		totalList.append(contents[x*2033:(x+1)*2033])
	return(totalList)

		#creates second array with specifically depression data which is in column 113 
	depList=list()
	for x in range(0,30000):
		depList.append(contents[(x*2033)+113])
	print(depList)
	return(depList)

def test_loader():
	file = open("data.txt","r") 
	contents=file.read()
		
		

		#print(contents)
		#print(contents)
		#creates array with each data set 
	totalList=list()
	for x in range(0, 30346):
		totalList.append(contents[x*2033:(x+1)*2033])

		#creates second array with specifically depression data which is in column 113 
	depList=list()
	for x in range(0,30346):
		depList.append(contents[(x*2033)+113])
	print(depList)




