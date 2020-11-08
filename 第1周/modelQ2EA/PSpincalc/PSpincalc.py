import numpy as np
import sys

# Jose Gama 2014
# Based on SpinCalc by John Fuller and SpinConv by Paolo de Leva.
# License: GPL (>= 3)
# A package for converting between attitude representations: DCM, Euler angles, Quaternions, and Euler vectors.
# Plus conversion between 2 Euler angle set types (xyx, yzy, zxz, xzx, yxy, zyz, xyz, yzx, zxy, xzy, yxz, zyx).
# Fully vectorized code, with warnings/errors for Euler angles (singularity, out of range, invalid angle order),
# DCM (orthogonality, not proper, exceeded tolerance to unity determinant) and Euler vectors(not unity).

############################## Qnorm

def Qnorm(Q):
    if type(Q) is list:
        Q=np.array(Q);
    elif type(Q) is tuple:
        Q=np.array(Q);
    if len(Q.shape)==1:
        if Q.shape[0] % 4 == 0:
            Q.shape=[Q.size//4,4]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    if Q.shape[1] != 4:
        Q.shape=[Q.size//4,4]
    Q=np.sqrt(np.power(Q,2).sum(axis=1))
    return(Q);

############################## Qnormalize
def Qnormalize(Q):
    if type(Q) is list:
        Q=np.array(Q);
    elif type(Q) is tuple:
        Q=np.array(Q);
    lqshp = len(Q.shape)
    if lqshp==1:
        if Q.shape[0] % 4 == 0:
            if Q.shape[0] > 4:
                Q.shape=[Q.size//4,4]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    elif Q.shape[lqshp-1] != 4:
        Q.shape=[Q.size//4,4]
    if lqshp==1:
        Q /= np.sqrt(np.power(Q,2).sum(axis=0))
    else:
        Q = (1/np.sqrt(np.power(Q,2).sum(axis=1)) * Q.T).T
    return(Q);

############################## EV2Q
def EV2Q(EV,tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # EV - [m1,m2,m3,MU] to Q - [q1,q2,q3,q4]
    # Euler vector (EV) and angle MU in radians
    if type(EV) is list:
        EV=np.array(EV);
    elif type(EV) is tuple:
        EV=np.array(EV);
    if len(EV.shape)==1:
        if EV.shape[0] % 4 == 0:
            EV.shape=[EV.size//4,4]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    if EV.shape[1] != 4:
        EV.shape=[EV.size//4,4]
    EVtmp = EV[:,0:4]#4
    halfMU = EV[:,3] / 2
    delta = np.sqrt(np.power(EVtmp, 2).sum(axis=0))
    if ignoreAllChk==False:
        if (abs(delta) > tol).any():
            print ("(At least one of the) input Euler vector(s) is not a unit vector\n")
            #sys.exit(1)
    # Quaternion
    SIN = np.sin(halfMU) # (Nx1)
    #Q = np.c_[ EVtmp[:,1]*SIN, EVtmp[:,2]*SIN, np.cos(halfMU), EVtmp[:,0]*SIN ]
    Q = np.c_[ np.cos(halfMU), EVtmp[:,0]*SIN, EVtmp[:,1]*SIN, EVtmp[:,2]*SIN ]
    return(Q);

############################## Q2EV
def Q2EV(Q,tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # Q - [q1,q2,q3,q4] to EV - [m1,m2,m3,MU]
    if type(Q) is list:
        Q=np.array(Q);
    elif type(Q) is tuple:
        Q=np.array(Q);
    if len(Q.shape)==1:
        if Q.shape[0] % 4 == 0:
            Q.shape=[Q.size//4,4]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    if Q.shape[1] != 4:
        Q.shape=[Q.size//4,4]
    if ~ignoreAllChk:
        if ichk and (abs(Q) > tol).any():
            print ("Warning: (At least one of the) Input quaternion(s) is not a unit vector\n")
    # Normalize quaternion(s) in case of deviation from unity.
    # User has already been warned of deviation.
    v_length=4; isnot_DCM=True;N=Q.shape[0]
    # Angle MU in radians and sine of MU/2
    Q2=Q[:,0:3]
    halfMUrad = np.arctan2( np.sqrt(np.power(Q2, 2).sum(axis=1)), Q[:,3] ) # Nx1
    SIN = np.sin(halfMUrad) # Nx1
    Zindex = np.where(SIN == 0) # [Nx1] Logical index
    if (np.array( Zindex )).size != 0 :
        # Initializing
        EV = np.zeros((N, 4))
        # Singular cases [MU is zero degrees]
        EV[Zindex, 0] = 1
        # Non-singular cases
        nZindex = np.where(SIN != 0)
        if (np.array( nZindex )).size != 0 :
            SIN = SIN[nZindex]
            EV[nZindex, :] = np.c_[ Q[nZindex,0] / SIN, Q[nZindex,1] / SIN, Q[nZindex,2] / SIN,(halfMUrad[nZindex] * 2).reshape(1,N) ]
    else:
        # Non-singular cases
        EV = np.c_[ Q[:,0]/SIN, Q[:,1]/SIN, Q[:,2]/SIN, halfMUrad * 2 ]
    # MU greater than 180 degrees
    Zindex = np.where(EV[:,3] > np.pi/2) # [Nx1] Logical index
    if (np.array( Zindex )).size != 0 :
        EV[Zindex, :] = np.c_[ (-EV[Zindex,0:3]).reshape((np.array( Zindex )).size,3), (2*np.pi-EV[Zindex,3]).reshape((np.array( Zindex )).size,1) ]
    return(EV);

############################## Q2DCM
def Q2DCM(Q,tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # Q - [q1,q2,q3,q4] to DCM - 3x3xN
    if type(Q) is list:
        Q=np.array(Q);
    elif type(Q) is tuple:
        Q=np.array(Q);
    if len(Q.shape)==1:
        if Q.shape[0] % 4 == 0:
            Q.shape=[Q.size//4,4]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    if Q.shape[1] != 4:
        Q.shape=[Q.size//4,4]
    if ~ignoreAllChk:
        if ichk and (abs(Q) > tol).any():
            print ("Warning: (At least one of the) Input quaternion(s) is not a unit vector\n")

    # Normalize quaternion(s) in case of deviation from unity.
    Qn = Qnormalize(Q)
    # User has already been warned of deviation.
    N=Q.shape[0]
    #Qn  = np.array(Qn).reshape(4, N)
    if N==1:
        DCM = np.array(np.zeros(9)).reshape(3, 3);
    else:
        DCM = 1.0 * np.array(range(9*N)).reshape(3, 3, N); # np.zeros(9*N)
    #if len(DCM.shape)==3:
    #    if DCM.shape==(3, 3, 1):
    #        DCM=DCM.reshape(3, 3)
    # DCM[0,0,:] = 1-2*(Qn[0,2,:]*Qn[0,2,:]+Qn[0,3,:]*Qn[0,3,:])
    # DCM[1,0,:] = 2*(Qn[0,1,:]*Qn[0,2,:]-Qn[0,0,:]*Qn[0,3,:])
    # DCM[2,0,:] = 2*(Qn[0,1,:]*Qn[0,3,:]+Qn[0,0,:]*Qn[0,2,:])
    # DCM[0,1,:] = 2*(Qn[0,1,:]*Qn[0,2,:]+Qn[0,0,:]*Qn[0,3,:])
    # DCM[1,1,:] = 1-2*(Qn[0,1,:]*Qn[0,1,:]+Qn[0,3,:]*Qn[0,3,:])
    # DCM[2,1,:] = 2*(Qn[0,2,:]*Qn[0,3,:]-Qn[0,0,:]*Qn[0,1,:])
    # DCM[0,2,:] = 2*(Qn[0,1,:]*Qn[0,3,:]-Qn[0,0,:]*Qn[0,2,:])
    # DCM[1,2,:] = 2*(Qn[0,2,:]*Qn[0,3,:]+Qn[0,0,:]*Qn[0,1,:])
    # DCM[2,2,:] = 1-2*(Qn[0,1,:]*Qn[0,1,:]+Qn[0,2,:]*Qn[0,2,:])
    if N==1:
        DCM[0,0] = 1-2*(Qn[0,2]*Qn[0,2]+Qn[0,3]*Qn[0,3])
        DCM[1,0] = 2*(Qn[0,1]*Qn[0,2]-Qn[0,0]*Qn[0,3])
        DCM[2,0] = 2*(Qn[0,1]*Qn[0,3]+Qn[0,0]*Qn[0,2])
        DCM[0,1] = 2*(Qn[0,1]*Qn[0,2]+Qn[0,0]*Qn[0,3])
        DCM[1,1] = 1-2*(Qn[0,1]*Qn[0,1]+Qn[0,3]*Qn[0,3])
        DCM[2,1] = 2*(Qn[0,2]*Qn[0,3]-Qn[0,0]*Qn[0,1])
        DCM[0,2] = 2*(Qn[0,1]*Qn[0,3]-Qn[0,0]*Qn[0,2])
        DCM[1,2] = 2*(Qn[0,2]*Qn[0,3]+Qn[0,0]*Qn[0,1])
        DCM[2,2] = 1-2*(Qn[0,1]*Qn[0,1]+Qn[0,2]*Qn[0,2])
    else:
        DCM[:,0,0] = 1-2*(Qn[:,2]*Qn[:,2]+Qn[:,3]*Qn[:,3])
        DCM[:,1,0] = 2*(Qn[:,1]*Qn[:,2]-Qn[:,0]*Qn[:,3])
        DCM[:,2,0] = 2*(Qn[:,1]*Qn[:,3]+Qn[:,0]*Qn[:,2])
        DCM[:,0,1] = 2*(Qn[:,1]*Qn[:,2]+Qn[:,0]*Qn[:,3])
        DCM[:,1,1] = 1-2*(Qn[:,1]*Qn[:,1]+Qn[:,3]*Qn[:,3])
        DCM[:,2,1] = 2*(Qn[:,2]*Qn[:,3]-Qn[:,0]*Qn[:,1])
        DCM[:,0,2] = 2*(Qn[:,1]*Qn[:,3]-Qn[:,0]*Qn[:,2])
        DCM[:,1,2] = 2*(Qn[:,2]*Qn[:,3]+Qn[:,0]*Qn[:,1])
        DCM[:,2,2] = 1-2*(Qn[:,1]*Qn[:,1]+Qn[:,2]*Qn[:,2])
    return(DCM);

############################## DCM2Q
def DCM2Q(DCM,tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # DCM - 3x3xN to Q - [q1,q2,q3,q4]
    # NOTE: Orthogonal matrixes may have determinant -1 or 1
    #       DCMs are special orthogonal matrices, with determinant 1
    if type(DCM) is list:
        DCM=np.array(DCM);
    elif type(DCM) is tuple:
        DCM=np.array(DCM);
    improper  = False
    DCM_not_1 = False
    if len(DCM.shape)>3 or len(DCM.shape)<2:
        print("DCM must be a 2-d or 3-d array.")
        sys.exit(1)
    if len(DCM.shape)==2:
        if DCM.size % 9 == 0:
            DCM.shape=[3,3]
        else:
            print ("Wrong number of elements1")
            sys.exit(1)
    if len(DCM.shape)==3:
        if np.prod(DCM.shape[1:3]) % 9 == 0:
            DCM.shape=[DCM.size//9,3,3]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    if len(DCM.shape)==2:
        N = 1
    else:
        N = DCM.shape[0]
    if N == 1:
        # Computing deviation from orthogonality
        delta = DCM.dot(DCM.T)- np.matrix(np.identity(3)) # DCM*DCM' - I
        delta = delta.reshape(9,1) # 9x1 <-- 3x3
        # Checking determinant of DCM
        DET = np.linalg.slogdet(DCM)[0]
        if DET<0:
            improper=True
        if ichk and np.abs(DET-1)>tol:
            DCM_not_1=True
        # Permuting  DCM
        DCM = DCM.reshape(1, 3, 3) # 1x3x3
    else:
        delta = np.zeros(DCM.shape)
        for x in range(N):
            delta[x,:,:] = DCM[x,:,:].dot(DCM[x,:,:].T) - np.matrix(np.identity(3));

        #dx = [ lambda x: DCM[:,:,x].dot(DCM[:,:,x].T) - np.matrix(np.identity(3)) for x in range(N) ]

        #delta = map(lambda x: DCM[:,:,x].dot(DCM[:,:,x].T) - np.matrix(np.identity(3)), range(N))
        #np.apply_along_axis(lambda x: DCM[:,:,x].dot(DCM[:,:,x].T) - np.matrix(np.identity(3)),2,DCM )

        #delta = np.array(dx)

        DET = DCM[:,0,0]*DCM[:,1,1]*DCM[:,2,2] -DCM[:,0,0]*DCM[:,1,2]*DCM[:,2,1]+DCM[:,0,1]*DCM[:,1,2]*DCM[:,2,0] -DCM[:,0,1]*DCM[:,1,0]*DCM[:,2,2]+DCM[:,0,2]*DCM[:,1,0]*DCM[:,2,1] -DCM[:,0,2]*DCM[:,1,1]*DCM[:,2,0]
        DET = DET.reshape(1, 1, N) # 1x1xN

        if (DET<0).any():
            improper=True
        if ichk and (np.abs(DET-1)>tol).any():
            DCM_not_1=True
        DCM2 = np.zeros(DCM.shape)

        #DCM2 <- vapply(1:N, function(cntDCM) DCM2[cntDCM,,] <- matrix(DCM[,,cntDCM],3,3), DCM2 )
        for x in range(N):
            DCM2[:,:,x]= DCM[:,x,:].T
        DCM = DCM2

    # Issuing error messages or warnings
    if ~ignoreAllChk:
        if ichk and (np.abs(delta)>tol).any():
            print("Warning: Input DCM is not orthogonal.")
    if ~ignoreAllChk:
        if improper:
            print("Improper input DCM")
            sys.exit(1)
    if ~ignoreAllChk:
        if DCM_not_1:
            print("Warning: Input DCM determinant off from 1 by more than tolerance.")

    # Denominators for 4 distinct types of equivalent Q equations
    if N==1:
        denom = np.c_[1.0 +  DCM[0,0,0] -  DCM[0,1,1] -  DCM[0,2,2], 1.0 -  DCM[0,0,0] +  DCM[0,1,1] -  DCM[0,2,2], 1.0 -  DCM[0,0,0] -  DCM[0,1,1] +  DCM[0,2,2], 1 +  DCM[0,0,0] +  DCM[0,1,1] +  DCM[0,2,2] ]
    else:
        denom = np.c_[1.0 +  DCM[0,:,0] -  DCM[1,:,1] -  DCM[2,:,2], 1.0 -  DCM[0,:,0] +  DCM[1,:,1] -  DCM[2,:,2], 1.0 -  DCM[0,:,0] -  DCM[1,:,1] +  DCM[2,:,2], 1 +  DCM[0,:,0] +  DCM[1,:,1] +  DCM[2,:,2] ]

    denom[np.where(denom<0)] = 0
    denom = 2 * np.sqrt(denom) # Nx4
    # Choosing for each DCM the equation which uses largest denominator
    maxdenom = denom.max(axis=1)
    #if len(maxdenom.shape) == 1:
    #    maxdenom = maxdenom.reshape(1,1)

    #if N==1:
    #    indexM = np.apply_along_axis(lambda x: np.where(x == denom.max(axis=1)) ,1,denom )
    #else:
    indexM = denom.argmax(axis=1)
        #indexM = np.apply_over_axes(lambda x,y: np.where(x == x.max(axis=1)) ,denom, axes=(0) )
        #indexM = np.apply_over_axes(lambda x: np.apply_along_axis(lambda y: np.where(y == denom.max(axis=1)) ,1,x )  ,denom, 1)

    Q = np.array(np.zeros(4*N)).reshape(N, 4) # Nx4
    if N==1:
        ii=0
        if indexM==0:
            Q = np.c_[ (DCM[ii,1,2]- DCM[ii,2,1]) / maxdenom, 0.25 * maxdenom, ( DCM[ii,0,1]+ DCM[ii,1,0]) / maxdenom,( DCM[ii,0,2]+ DCM[ii,2,0]) / maxdenom]
        if indexM==1:
            Q = np.c_[ (DCM[ii,2,0]- DCM[ii,0,2]) / maxdenom,( DCM[ii,0,1]+ DCM[ii,1,0]) / maxdenom,0.25 * maxdenom,( DCM[ii,1,2]+ DCM[ii,2,1]) / maxdenom]
        if indexM==2:
            Q = np.c_[ (DCM[ii,0,1]- DCM[ii,1,0]) / maxdenom,( DCM[ii,0,2]+ DCM[ii,2,0]) / maxdenom,( DCM[ii,1,2]+ DCM[ii,2,1]) / maxdenom,0.25 * maxdenom]
        if indexM==3:
            Q = np.c_[0.25 * maxdenom,( DCM[ii,1,2]- DCM[ii,2,1]) / maxdenom,( DCM[ii,2,0]- DCM[ii,0,2]) / maxdenom,( DCM[ii,0,1]- DCM[ii,1,0]) / maxdenom]
        return(Q);
    else:
        ii= np.where(indexM == 0)[0]
        if len(ii)>0:
            Q[ii,:] = np.c_[ -(DCM[1,ii,2]- DCM[2,ii,1]) / maxdenom[ii], 0.25 * maxdenom[ii], ( DCM[0,ii,1]+ DCM[1,ii,0]) / maxdenom[ii],( DCM[0,ii,2]+ DCM[2,ii,0]) / maxdenom[ii] ]
        ii= np.where(indexM == 1)[0]
        if len(ii)>0:
            Q[ii,:] = np.c_[ -(DCM[2,ii,0]- DCM[0,ii,2]) / maxdenom[ii],( DCM[0,ii,1]+ DCM[1,ii,0]) / maxdenom[ii],0.25 * maxdenom[ii],( DCM[1,ii,2]+ DCM[2,ii,1]) / maxdenom[ii] ]
        ii= np.where(indexM == 2)[0]
        if len(ii)>0:
            Q[ii,:] = np.c_[ (DCM[0,ii,1]- DCM[1,ii,0]) / maxdenom[ii],( DCM[0,ii,2]+ DCM[2,ii,0]) / maxdenom[ii],( DCM[1,ii,2]+ DCM[2,ii,1]) / maxdenom[ii],0.25 * maxdenom[ii] ]
        ii= np.where(indexM == 3)[0]
        if len(ii)>0:
            Q[ii,:] = np.c_[0.25 * maxdenom[ii],-( DCM[1,ii,2]- DCM[2,ii,1]) / maxdenom[ii],-( DCM[2,ii,0]- DCM[0,ii,2]) / maxdenom[ii],-( DCM[0,ii,1]- DCM[1,ii,0]) / maxdenom[ii] ]
        return(Q);


############################## EA2EV
def EA2EV(EA,EulerOrder="zyx",tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # EA - [psi,theta,phi] to EV - [m1,m2,m3,MU]
    if (EulerOrder in ["zyx","zxy","yxz","xzy","xyz","yzx","zyz","zxz","yxy","yzy","xyx","xzx"])==False:
        print("Invalid input Euler angle order")
        sys.exit(1)
    if type(EA) is list:
        EA=np.array(EA);
    elif type(EA) is tuple:
        EA=np.array(EA);
    if len(EA.shape)==1:
        if EA.shape[0] % 3 == 0:
            EA.shape=[EA.size//3,3]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    if EA.shape[1] != 3:
        EA.shape=[EA.size//3,3]
    Q=EA2Q(EA, EulerOrder,tol, ichk, ignoreAllChk);
    EV=Q2EV(Q, tol, ichk, ignoreAllChk);
    return(EV);

############################## EV2EA
def EV2EA(EV,EulerOrder="zyx",tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # EV - [m1,m2,m3,MU] to EA - [psi,theta,phi]
    if (EulerOrder in ["zyx","zxy","yxz","xzy","xyz","yzx","zyz","zxz","yxy","yzy","xyx","xzx"])==False:
        print("Invalid input Euler angle order")
        sys.exit(1)
    if type(EV) is list:
        EV=np.array(EV);
    elif type(EV) is tuple:
        EV=np.array(EV);
    if len(EV.shape)==1:
        if EV.shape[0] % 4 == 0:
            EV.shape=[EV.size//4,4]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    if EV.shape[1] != 4:
        EV.shape=[EV.size//4,4]
    Q=EV2Q(EV, tol, ichk, ignoreAllChk);
    EA=Q2EA(Q, EulerOrder, tol, ichk, ignoreAllChk);
    return(EA);



############################## EA2Q
def EA2Q(EA,EulerOrder="zyx",tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # EA - [psi,theta,phi] yaw, pitch, roll to EV - [m1,m2,m3,MU]
    # EA in radians
    # ichk = FALSE disables near-singularity warnings.
    # Identify singularities (2nd Euler angle out of range)
    if (EulerOrder in ["zyx","zxy","yxz","xzy","xyz","yzx","zyz","zxz","yxy","yzy","xyx","xzx"])==False:
        print("Invalid input Euler angle order")
        sys.exit(1)
    if type(EA) is list:
        EA=np.array(EA);
    elif type(EA) is tuple:
        EA=np.array(EA);
    if len(EA.shape)==1:
        if EA.shape[0] % 3 == 0:
            EA.shape=[EA.size//3,3]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    if EA.shape[1] != 3:
        EA.shape=[EA.size//3,3]
    theta = EA[:, 1] # Nx1
    if ignoreAllChk==False:
        if EulerOrder[0]!=EulerOrder[2]:
            # Type 1 rotation about three distinct axes
            if (abs(theta) >= np.pi/2 ).any():
                print("Second input Euler angle(s) outside -90 to 90 degree range")
                sys.exit(1)
            elif ichk and (abs(theta)>88 * (np.pi/180)).any():
                print('Warning: Second input Euler angle(s) near a singularity (-90 or 90 degrees).')
        else:
            # Type 2 rotation (1st and 3rd rotation about same axis)
            if ((theta<=0).any() or (theta >= np.pi).any()):
                print("Second input Euler angle(s) outside 0 to 180 degree range")
                sys.exit(1)
            elif ichk and ((theta < 2 * (np.pi/180)).any() or (theta>178 * (np.pi/180))):
                print("Warning: Second input Euler angle(s) near a singularity (0 or 180 degrees).")
    # Half angles in radians
    HALF = EA / 2# * (pi/360) # Nx3
    Hpsi   = HALF[:,0] # Nx1
    Htheta = HALF[:,1] # Nx1
    Hphi   = HALF[:,2] # Nx1
    # Pre-calculate cosines and sines of the half-angles for conversion.
    c1=np.cos(Hpsi); c2=np.cos(Htheta); c3=np.cos(Hphi)
    s1=np.sin(Hpsi); s2=np.sin(Htheta); s3=np.sin(Hphi)
    c13 =np.cos(Hpsi+Hphi);  s13 =np.sin(Hpsi+Hphi)
    c1_3=np.cos(Hpsi-Hphi);  s1_3=np.sin(Hpsi-Hphi)
    c3_1=np.cos(Hphi-Hpsi);  s3_1=np.sin(Hphi-Hpsi)
    if EulerOrder=="xyx":
        Q=np.c_[c2*c13, c2*s13,s2*c1_3, s2*s1_3]
    elif EulerOrder=="yzy":
        Q=np.c_[c2*c13, s2*s1_3,c2*s13, s2*c1_3]
    elif EulerOrder=="zxz":
        Q=np.c_[c2*c13, s2*c1_3,s2*s1_3, c2*s13]
    elif EulerOrder=="xzx":
        Q=np.c_[c2*c13, c2*s13,s2*s3_1, s2*c3_1]
    elif EulerOrder=="yxy":
        Q=np.c_[c2*c13, s2*c3_1,c2*s13,  s2*s3_1]
    elif EulerOrder=="zyz":
        Q=np.c_[c2*c13, s2*s3_1,s2*c3_1, c2*s13]
    elif EulerOrder=="xyz":
        Q=np.c_[c1*c2*c3-s1*s2*s3, s1*c2*c3+c1*s2*s3,c1*s2*c3-s1*c2*s3, c1*c2*s3+s1*s2*c3]
    elif EulerOrder=="yzx":
        Q=np.c_[c1*c2*c3-s1*s2*s3, c1*c2*s3+s1*s2*c3,s1*c2*c3+c1*s2*s3, c1*s2*c3-s1*c2*s3]
    elif EulerOrder=="zxy":
        Q=np.c_[c1*c2*c3-s1*s2*s3, c1*s2*c3-s1*c2*s3,c1*c2*s3+s1*s2*c3, s1*c2*c3+c1*s2*s3]
    elif EulerOrder=="xzy":
        Q=np.c_[c1*c2*c3+s1*s2*s3, s1*c2*c3-c1*s2*s3,c1*c2*s3-s1*s2*c3, c1*s2*c3+s1*c2*s3]
    elif EulerOrder=="yxz":
        Q=np.c_[c1*c2*c3+s1*s2*s3, c1*s2*c3+s1*c2*s3,s1*c2*c3-c1*s2*s3, c1*c2*s3-s1*s2*c3]
    elif EulerOrder=="zyx":
        Q=np.c_[c1*c2*c3+s1*s2*s3, c1*c2*s3-s1*s2*c3,c1*s2*c3+s1*c2*s3, s1*c2*c3-c1*s2*s3]
    elif ignoreAllChk==False:
        print("Invalid input Euler angle order")
        sys.exit(1)
    return(Q)

############################## Q2EA
def Q2EA(Q,EulerOrder="zyx",tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # Implementation of quaternion to Euler angles based on D. M. Henderson 1977
    # Shuttle Program. Euler Angles, Quaternions, and Transformation Matrices Working Relationships.
    # National Aeronautics and Space Administration (NASA), N77-31234/6
    # Q - [q1,q2,q3,q4] to EA - [psi,theta,phi]
    # Madgwick (zyx) originaly used Q = [phi, theta, psi]
    # Jose Gama 2014
    if type(Q) is list:
        Q=np.array(Q);
    elif type(Q) is tuple:
        Q=np.array(Q);
    if len(Q.shape)==1:
        if Q.shape[0] % 4 == 0:
            Q.shape=[Q.size//4,4]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    if Q.shape[1] != 4:
        Q.shape=[Q.size//4,4]
    if ~ignoreAllChk:
        if ichk and (abs(Q) > tol).any():
            print ("Warning: (At least one of the) Input quaternion(s) is not a unit vector\n")
    # Normalize quaternion(s) in case of deviation from unity.
    Qn = Qnormalize(Q)
    if (EulerOrder in ["zyx","zxy","yxz","xzy","xyz","yzx","zyz","zxz","yxy","yzy","xyx","xzx"])==False:
        print("Invalid input Euler angle order")
        sys.exit(1)
    N=Q.shape[0]
    if ignoreAllChk==False:
        if ichk and (abs(np.sqrt(np.power(Q,2).sum(axis=1) - 1)) > tol).any():
            print("Warning: (At least one of the) Input quaternion(s) is not a unit vector")
    if EulerOrder=="zyx":
        EA = np.c_[np.arctan2((2*(Q[:,1]*Q[:,2] + Q[:,0]*Q[:,3])),(np.power(Q[:,0],2) + np.power(Q[:,1],2) - np.power(Q[:,2],2) - np.power(Q[:,3],2))), np.arctan2(-(2*(Q[:,1]*Q[:,3] - Q[:,0]*Q[:,2])),np.sqrt(1-np.power(2*(Q[:,1]*Q[:,3] - Q[:,0]*Q[:,2]),2))),np.arctan2((2*(Q[:,2]*Q[:,3] + Q[:,0]*Q[:,1])),(np.power(Q[:,0],2) - np.power(Q[:,1],2) - np.power(Q[:,2],2) + np.power(Q[:,3],2)))]
    elif EulerOrder=="yxz":
        EA = np.c_[np.arctan2(2*(Q[:,1]*Q[:,3] + Q[:,0]*Q[:,2]), 1-2*(np.power(Q[:,1],2) + np.power(Q[:,2],2))), np.arctan2(-(2*(Q[:,2]*Q[:,3] - Q[:,0]*Q[:,1])),np.sqrt(1-np.power(2*(Q[:,2]*Q[:,3] - Q[:,0]*Q[:,1]),2))),np.arctan2((2*(Q[:,1]*Q[:,2] + Q[:,0]*Q[:,3])),(np.power(Q[:,0],2) - np.power(Q[:,1],2) + np.power(Q[:,2],2)- np.power(Q[:,3],2)))]
    elif EulerOrder=="xzy":
        EA = np.c_[ - np.arctan2(-(2*(Q[:,2]*Q[:,3] + Q[:,0]*Q[:,1])),(np.power(Q[:,0],2) - np.power(Q[:,1],2) + np.power(Q[:,2],2)- np.power(Q[:,3],2))), np.arctan2(-(2*(Q[:,1]*Q[:,2] - Q[:,0]*Q[:,3])),np.sqrt(1-np.power(2*(Q[:,1]*Q[:,2] - Q[:,0]*Q[:,3]),2))),np.arctan2((2*(Q[:,1]*Q[:,3] + Q[:,0]*Q[:,2])),(np.power(Q[:,0],2) + np.power(Q[:,1],2) - np.power(Q[:,2],2) - np.power(Q[:,3],2)))]
    elif EulerOrder=="zxy":
        EA = np.c_[np.arctan2(-(2*(Q[:,1]*Q[:,2] - Q[:,0]*Q[:,3])),(np.power(Q[:,0],2) - np.power(Q[:,1],2) + np.power(Q[:,2],2)- np.power(Q[:,3],2))), np.arctan2((2*(Q[:,2]*Q[:,3] + Q[:,0]*Q[:,1])),np.sqrt(1-np.power(2*(Q[:,2]*Q[:,3] + Q[:,0]*Q[:,1]),2))),np.arctan2(-(2*(Q[:,1]*Q[:,3] - Q[:,0]*Q[:,2])),(np.power(Q[:,0],2) - np.power(Q[:,1],2) - np.power(Q[:,2],2) + np.power(Q[:,3],2)))]
    elif EulerOrder=="yzx":
        EA = np.c_[np.arctan2(-(2*(Q[:,1]*Q[:,3] - Q[:,0]*Q[:,2])),(np.power(Q[:,0],2) + np.power(Q[:,1],2) - np.power(Q[:,2],2) - np.power(Q[:,3],2))), np.arctan2((2*(Q[:,1]*Q[:,2] + Q[:,0]*Q[:,3])),np.sqrt(1-np.power(2*(Q[:,1]*Q[:,2] + Q[:,0]*Q[:,3]),2))),np.arctan2(-(2*(Q[:,2]*Q[:,3] - Q[:,0]*Q[:,1])),(np.power(Q[:,0],2) - np.power(Q[:,1],2) + np.power(Q[:,2],2)- np.power(Q[:,3],2)))]
    elif EulerOrder=="xyz":
        EA = np.c_[np.arctan2(-(2*(Q[:,2]*Q[:,3] - Q[:,0]*Q[:,1])),(np.power(Q[:,0],2) - np.power(Q[:,1],2) - np.power(Q[:,2],2) + np.power(Q[:,3],2))), np.arctan2((2*(Q[:,1]*Q[:,3] + Q[:,0]*Q[:,2])),np.sqrt(1-np.power(2*(Q[:,1]*Q[:,3] + Q[:,0]*Q[:,2]),2))),np.arctan2(-(2*(Q[:,1]*Q[:,2] - Q[:,0]*Q[:,3])),(np.power(Q[:,0],2) + np.power(Q[:,1],2) - np.power(Q[:,2],2) - np.power(Q[:,3],2)))]
    elif EulerOrder=="zyz":
        EA = np.c_[np.arctan2((2*(Q[:,2]*Q[:,3] - Q[:,0]*Q[:,1])),(2*(Q[:,1]*Q[:,3] + Q[:,0]*Q[:,2]))), np.arctan2(np.sqrt(1-np.power(np.power(Q[:,0],2) - np.power(Q[:,1],2) - np.power(Q[:,2],2) + np.power(Q[:,3],2),2)),(np.power(Q[:,0],2) - np.power(Q[:,1],2) - np.power(Q[:,2],2) + np.power(Q[:,3],2))),np.arctan2((2*(Q[:,2]*Q[:,3] + Q[:,0]*Q[:,1])),-(2*(Q[:,1]*Q[:,3] - Q[:,0]*Q[:,2])))]
    elif EulerOrder=="zxz":
        EA = np.c_[np.arctan2((2*(Q[:,1]*Q[:,3] + Q[:,0]*Q[:,2])),-(2*(Q[:,2]*Q[:,3] - Q[:,0]*Q[:,1]))), np.arctan2(np.sqrt(1-np.power(np.power(Q[:,0],2) - np.power(Q[:,1],2) - np.power(Q[:,2],2) + np.power(Q[:,3],2),2)),(np.power(Q[:,0],2) - np.power(Q[:,1],2) - np.power(Q[:,2],2) + np.power(Q[:,3],2))),np.arctan2((2*(Q[:,1]*Q[:,3] - Q[:,0]*Q[:,2])),(2*(Q[:,2]*Q[:,3] + Q[:,0]*Q[:,1])))]
    elif EulerOrder=="yxy":
        EA = np.c_[np.arctan2((2*(Q[:,1]*Q[:,2] - Q[:,0]*Q[:,3])),(2*(Q[:,2]*Q[:,3] + Q[:,0]*Q[:,1]))), np.arctan2(np.sqrt(1-np.power(np.power(Q[:,0],2) - np.power(Q[:,1],2) + np.power(Q[:,2],2)- np.power(Q[:,3],2),2)),(np.power(Q[:,0],2) - np.power(Q[:,1],2) + np.power(Q[:,2],2)- np.power(Q[:,3],2))),np.arctan2((2*(Q[:,1]*Q[:,2] + Q[:,0]*Q[:,3])),-(2*(Q[:,2]*Q[:,3] - Q[:,0]*Q[:,1])))]
    elif EulerOrder=="yzy":
        EA = np.c_[np.arctan2((2*(Q[:,2]*Q[:,3] + Q[:,0]*Q[:,1])),-(2*(Q[:,1]*Q[:,2] - Q[:,0]*Q[:,3]))), np.arctan2(np.sqrt(1-np.power(np.power(Q[:,0],2) - np.power(Q[:,1],2) + np.power(Q[:,2],2)- np.power(Q[:,3],2),2)),(np.power(Q[:,0],2) - np.power(Q[:,1],2) + np.power(Q[:,2],2)- np.power(Q[:,3],2))),np.arctan2((2*(Q[:,2]*Q[:,3] - Q[:,0]*Q[:,1])),(2*(Q[:,1]*Q[:,2] + Q[:,0]*Q[:,3])))]
    elif EulerOrder=="xzx":
        EA = np.c_[np.arctan2((2*(Q[:,1]*Q[:,3] - Q[:,0]*Q[:,2])),(2*(Q[:,1]*Q[:,2] + Q[:,0]*Q[:,3]))), np.arctan2(np.sqrt(1-np.power(np.power(Q[:,0],2) + np.power(Q[:,1],2) - np.power(Q[:,2],2) - np.power(Q[:,3],2),2)),(np.power(Q[:,0],2) + np.power(Q[:,1],2) - np.power(Q[:,2],2) - np.power(Q[:,3],2))),np.arctan2((2*(Q[:,1]*Q[:,3] + Q[:,0]*Q[:,2])),-(2*(Q[:,1]*Q[:,2] - Q[:,0]*Q[:,3])))]
    elif EulerOrder=="xyx":
        EA = np.c_[np.arctan2((2*(Q[:,1]*Q[:,2] + Q[:,0]*Q[:,3])),-(2*(Q[:,1]*Q[:,3] - Q[:,0]*Q[:,2]))), np.arctan2(np.sqrt(1-np.power(np.power(Q[:,0],2) + np.power(Q[:,1],2) - np.power(Q[:,2],2) - np.power(Q[:,3],2),2)),(np.power(Q[:,0],2) + np.power(Q[:,1],2) - np.power(Q[:,2],2) - np.power(Q[:,3],2))),np.arctan2((2*(Q[:,1]*Q[:,2] - Q[:,0]*Q[:,3])),(2*(Q[:,1]*Q[:,3] + Q[:,0]*Q[:,2])))]
    else:
        print("Invalid input Euler angle order")
        sys.exit(1)
    #EA = EA * (180/pi) # (Nx3) Euler angles in degrees
    theta  = EA[:,1]       # (Nx1) Angle THETA in degrees
    # Check EA
    if ignoreAllChk==False:
        if isinstance(EA, complex):
            print("Unreal\nUnreal Euler EA. Input resides too close to singularity.\nPlease choose different EA type.")
            sys.exit(1)
        # Type 1 rotation (rotations about three distinct axes)
        # THETA is computed using ASIN and ranges from -90 to 90 degrees

    if ignoreAllChk==False:
        if EulerOrder[0] != EulerOrder[2]:
            singularities = np.abs(theta) > 89.9*(np.pi/180) # (Nx1) Logical index
            singularities[np.where(np.isnan(singularities))] = False
            if len(singularities)>0:
                if (singularities).any():
                    firstsing = np.where(singularities)[0] #which(singularities)[1] # (1x1)
                    print("Input rotation ", firstsing, " resides too close to Type 1 Euler singularity.\nType 1 Euler singularity occurs when second angle is -90 or 90 degrees.\nPlease choose different EA type.")
                    sys.exit(1)
        else:
            # Type 2 rotation (1st and 3rd rotation about same axis)
            # THETA is computed using ACOS and ranges from 0 to 180 degrees
            singularities = (theta<0.1*(np.pi/180)) | (theta>179.9*(np.pi/180)) # (Nx1) Logical index
            singularities[np.where(np.isnan(singularities))] = False
            if (len(singularities)>0):
                if((singularities).any()):
                    firstsing = np.where(singularities)[0] # (1x1)
                    print("Input rotation ", firstsing, " resides too close to Type 2 Euler singularity.\nType 2 Euler singularity occurs when second angle is 0 or 180 degrees.\nPlease choose different EA type.")
                    sys.exit(1)
    return(EA)

def DCM2EA(DCM, EulerOrder="zyx", tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # DCM - 3x3xN to EA -
    # NOTE: Orthogonal matrixes may have determinant -1 or 1
    #       DCMs are special orthogonal matrices, with determinant 1
    if (EulerOrder in ["zyx","zxy","yxz","xzy","xyz","yzx","zyz","zxz","yxy","yzy","xyx","xzx"])==False:
        print("Invalid input Euler angle order")
        sys.exit(1)
    if type(DCM) is list:
        DCM=np.array(DCM);
    elif type(DCM) is tuple:
        DCM=np.array(DCM);
    improper  = False
    DCM_not_1 = False
    if len(DCM.shape)>3 or len(DCM.shape)<2:
        print("DCM must be a 2-d or 3-d array.")
        sys.exit(1)
    if len(DCM.shape)==2:
        if DCM.size % 9 == 0:
            DCM.shape=[3,3]
        else:
            print ("Wrong number of elements1")
            sys.exit(1)
    if len(DCM.shape)==3:
        if np.prod(DCM.shape[1:3]) % 9 == 0:
            DCM.shape=[DCM.size//9,3,3]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    Q=DCM2Q(DCM, tol, ichk, ignoreAllChk)
    EA=Q2EA(Q, EulerOrder, tol, ichk, ignoreAllChk);
    return(EA);

def DCM2EV(DCM, tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # DCM - 3x3xN to EV -
    # NOTE: Orthogonal matrixes may have determinant -1 or 1
    #       DCMs are special orthogonal matrices, with determinant 1
    if type(DCM) is list:
        DCM=np.array(DCM);
    elif type(DCM) is tuple:
        DCM=np.array(DCM);
    improper  = False
    DCM_not_1 = False
    if len(DCM.shape)>3 or len(DCM.shape)<2:
        print("DCM must be a 2-d or 3-d array.")
        sys.exit(1)
    if len(DCM.shape)==2:
        if DCM.size % 9 == 0:
            DCM.shape=[3,3]
        else:
            print ("Wrong number of elements1")
            sys.exit(1)
    if len(DCM.shape)==3:
        if np.prod(DCM.shape[1:3]) % 9 == 0:
            DCM.shape=[DCM.size//9,3,3]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    Q=DCM2Q(DCM, tol, ichk, ignoreAllChk)
    EV=Q2EV(Q, tol, ichk, ignoreAllChk);
    return(EV);

def EV2DCM(EV, tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # EV - [m1,m2,m3,MU] to DCM
    if type(EV) is list:
        EV=np.array(EV);
    elif type(EV) is tuple:
        EV=np.array(EV);
    if len(EV.shape)==1:
        if EV.shape[0] % 4 == 0:
            EV.shape=[EV.size//4,4]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    if EV.shape[1] != 4:
        EV.shape=[EV.size//4,4]
    Q=EV2Q(EV, tol, ichk, ignoreAllChk);
    DCM=Q2DCM(Q, tol, ichk, ignoreAllChk);
    return(DCM);

def EA2DCM(EA,EulerOrder="zyx",tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # EA - [psi,theta,phi] yaw, pitch, roll to DCM
    # EA in radians
    # ichk = FALSE disables near-singularity warnings.
    # Identify singularities (2nd Euler angle out of range)
    if (EulerOrder in ["zyx","zxy","yxz","xzy","xyz","yzx","zyz","zxz","yxy","yzy","xyx","xzx"])==False:
        print("Invalid input Euler angle order")
        sys.exit(1)
    if type(EA) is list:
        EA=np.array(EA);
    elif type(EA) is tuple:
        EA=np.array(EA);
    if len(EA.shape)==1:
        if EA.shape[0] % 3 == 0:
            EA.shape=[EA.size//3,3]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    if EA.shape[1] != 3:
        EA.shape=[EA.size//3,3]
    Q=EA2Q(EA,EulerOrder, tol, ichk, ignoreAllChk);
    DCM=Q2DCM(Q, tol, ichk, ignoreAllChk);
    return(DCM);

def EA2EA(EA,EulerOrder1="zyx",EulerOrder2="xyz",tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # EA - [psi,theta,phi] yaw, pitch, roll to DCM
    # EA in radians
    # ichk = FALSE disables near-singularity warnings.
    # Identify singularities (2nd Euler angle out of range)
    if (EulerOrder1 in ["zyx","zxy","yxz","xzy","xyz","yzx","zyz","zxz","yxy","yzy","xyx","xzx"])==False:
        print("Invalid input Euler angle order")
        sys.exit(1)
    if (EulerOrder2 in ["zyx","zxy","yxz","xzy","xyz","yzx","zyz","zxz","yxy","yzy","xyx","xzx"])==False:
        print("Invalid input Euler angle order")
        sys.exit(1)
    if type(EA) is list:
        EA=np.array(EA);
    elif type(EA) is tuple:
        EA=np.array(EA);
    if len(EA.shape)==1:
        if EA.shape[0] % 3 == 0:
            EA.shape=[EA.size//3,3]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    if EA.shape[1] != 3:
        EA.shape=[EA.size//3,3]
    Q=EA2Q(EA,EulerOrder1, tol, ichk, ignoreAllChk);
    Q=Qnormalize(Q);
    EA2=Q2EA(Q,EulerOrder2, tol, ichk, ignoreAllChk);
    return(EA2);

def Q2GL(Q, tol = 10 * np.spacing(1), ichk=False, ignoreAllChk=False):
    # Q - [q1,q2,q3,q4] to DCM - 3x3xN
    if type(Q) is list:
        Q=np.array(Q);
    elif type(Q) is tuple:
        Q=np.array(Q);
    if len(Q.shape)==1:
        if Q.shape[0] % 4 == 0:
            Q.shape=[Q.size//4,4]
        else:
            print ("Wrong number of elements")
            sys.exit(1)
    if Q.shape[1] != 4:
        Q.shape=[Q.size//4,4]
    if ~ignoreAllChk:
        if ichk and (abs(Q) > tol).any():
            print ("Warning: (At least one of the) Input quaternion(s) is not a unit vector\n")
    # Normalize quaternion(s) in case of deviation from unity.
    Q = Qnormalize(Q)
    N=Q.shape[0]
    GL = np.zeros((Q.shape[0]*4,4))
    for n in range(N):
        x=Q[n,0]
        y=Q[n,1]
        w=Q[n,2]
        z=Q[n,3]
        #GL[n*4,:] = [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y), 0];
        #GL[n*4+1,:] = [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x), 0];
        #GL[n*4+2,:] = [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y), 0];
        GL[n*4,:] = [1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y), 0];
        GL[n*4+1,:] = [2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x), 0];
        GL[n*4+2,:] = [2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y), 0];

        GL[n*4+3,:] = [0, 0, 0, 1];
    return(GL);
