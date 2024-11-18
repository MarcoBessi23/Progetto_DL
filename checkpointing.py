import numpy as np






#Algorithm taken from REVOLVE paper
while(action != "terminate"):
    check , capo, fine = -1, 0, steps
    snaps = adjust(steps)
    oldcapo = capo
    action = revolve(check, capo, fine, snaps, info)
    if action == "advance":
        #advance the system to the state capo 
        for j in range(oldcapo, capo):
            forward(w)
            #break
    elif action == "takeshot":
        #saving current state as a step
        for i in range(n):
            ustor[check][i] = u[i];
            #break;           
    elif action == "reverse":
        #Advance with recording and perform first reverse step
        for i in range(n): #Initialize adjoints 
            bu[i]= u[i]-ustar[i] 
            bz[i]=0
            reverse(bu,bz) # First reverse step 
            #break;
    # Advance with recording and perform first reverse step 
    elif action == "firstrun": 
        forwardrec(u)
        for i in range(n): # Initialize adjoints 
            bu[i] = u[i]-ustar[i] 
            bz[i] = 0
            reverse(bu,bz) # First reverse step 
            #break;## Subsequent, combined forward/reverse steps
    elif action == "youturn": 
            forwardrec(u);
            reverse(bu,bz);
    elif action == "restore": 
        for i in range(n):
            u[i] = ustor[check][i];
    elif action == "error":
            print("scheduling error")
            exit(-1);