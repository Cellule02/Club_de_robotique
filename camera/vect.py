

############ 300 cm #############
#                               # 2
#        QR3          QR4       # 0
#                               # 0
#        QR1          QR2       # c
#(0,0)                          # m
#################################
#00

# Position Images Virtuelle Aplati
QR1_Virt = (196,461) # en pxl
QR2_Virt = (676,271) # en pxl
QR3_Virt = (256,741) # en pxl
QR4_Virt = (736,551) # en pxl
QR_Robot_Virt = (412,619) # en pxl




def add(X,Y):
	return (X[0]+Y[0], X[1]+Y[1])

def moins(X,Y):
	return (X[0]-Y[0], X[1]-Y[1])

def mul(r,X):
	return (X[0]*r, X[1]*r)

def dist(X,Y):
	return int(((X[0]-Y[0])**2 + (X[1]-Y[1])**2 ) **0.5)


def corrigeDeformation(Pt_QR1_Virt,Pt_QR2_Virt,Pt_QR3_Virt,Pt_QR4_Virt,Pt_Interet_Virt):
	# Pt_QR1_Virt Position (X:int ,Y:int) du QRCODE 1 sur l'image (en Pxl)
	# Pt_QR2_Virt Position (X:int ,Y:int) du QRCODE 2 sur l'image (en Pxl)
	# Pt_QR3_Virt Position (X:int ,Y:int) du QRCODE 3 sur l'image (en Pxl)
	# Pt_QR4_Virt Position (X:int ,Y:int) du QRCODE 4 sur l'image (en Pxl)
	# Pt_Interet_Virt Position (X:int ,Y:int) du point d'interet Virtuel

	Pt_QR1_Reel = (50 , 50 ) # Pt_QR1_Virt Position (X:int ,Y:int) du QRCODE 1 sur le terrain (en cm)
	Pt_QR2_Reel = (250, 50 ) # Pt_QR2_Virt Position (X:int ,Y:int) du QRCODE 2 sur le terrain (en cm)
	Pt_QR3_Reel = (50 , 150) # Pt_QR3_Virt Position (X:int ,Y:int) du QRCODE 3 sur le terrain (en cm)
	Pt_QR4_Reel = (250, 150) # Pt_QR4_Virt Position (X:int ,Y:int) du QRCODE 4 sur le terrain (en cm)
	U_Reel = moins(Pt_QR2_Reel, Pt_QR1_Reel)
	V_Reel = moins(Pt_QR3_Reel, Pt_QR1_Reel)

	U_Virt = moins(Pt_QR2_Virt, Pt_QR1_Virt)
	V_Virt = moins(Pt_QR3_Virt, Pt_QR1_Virt)

	###########################################################################
	##### Verification de la précision en usant le 4eme QrCode Comme Ref  #####
	Pt_QR4_Virt_Theorique = add(add(U_Virt, V_Virt), Pt_QR1_Virt)
	diffQR4 = dist(Pt_QR4_Virt_Theorique,Pt_QR4_Virt)
	if diffQR4 >= 50 : # Seuil en Pxl
		raise ValueError("Erreur trop importante ("+ str(diffQR4) +" pxl)")

	print("Précision : DiffQR4 =",diffQR4,"pxl")
	###########################################################################
	###########################################################################

	VectPtInteret_Virt = moins(Pt_Interet_Virt, Pt_QR1_Virt)


	###########################################################################
	######                          Explication                          ######
	# On veut CoefX et CoefY tel que :
	# VectPtInteret_Virt[0] = CoefX * U_Virt[0] + CoefY * V_Virt[0] 
	# VectPtInteret_Virt[1] = CoefX * U_Virt[1] + CoefY * V_Virt[1] 
	# 
	# Pour résoudre ce systéme on utilise la méthode de Cramer (cf wiki)
	# On a le code pour résoudre le systeme suivant :
	# e1[2] =  e1[0] * CoefX + e1[1] * CoefY
	# e2[2] =  e2[0] * CoefX + e2[1] * CoefY

	# on reconstruit nos 2 équations :
	e1 = [ U_Virt[0], V_Virt[0], VectPtInteret_Virt[0] ]
	e2 = [ U_Virt[1], V_Virt[1], VectPtInteret_Virt[1] ]

	determinant=e1[0]*e2[1]-e1[1]*e2[0]
	if determinant==0:
		raise ValueError("Vecteurs Colinéaires")
	else:
		CoefX = (e1[2]*e2[1]-e1[1]*e2[2])/determinant
		CoefY = (e1[0]*e2[2]-e1[2]*e2[0])/determinant

	###########################################################################
	###########################################################################

	# on utilise les 2 coefficients pour l'utiliser sur nos vecteurs Réels
	Pt_Interet_Reel = add(Pt_QR1_Reel,add(mul(CoefX,U_Reel),mul(CoefY,V_Reel)))

	#return Pt_Interet_Reel # sans arrondie
	return (int(Pt_Interet_Reel[0]),int(Pt_Interet_Reel[1])) # avec arrondi au cm 


Pt_robot = corrigeDeformation(QR1_Virt,QR2_Virt,QR3_Virt,QR4_Virt,QR_Robot_Virt)

print(Pt_robot) 
