   PROGRAM ODFTEX_2ORDERSTEP    
!  Calculates evolution of ODF with active drx (new one-parameter model)
!  Accepts an arbitrary velocity gradient tensor (assumed constant in time)
!  Operative slip system assumed (only one for now):
!     s=1: (010)[100] (olivine)
!  The limitation to one slip system means that the only nonzero spin is \dot\psi
!  Time is scaled by 1/Sqrt[E_{ij} E_{ij}/2]
!  Second-order accurate midpoint time stepping impemented

!  Input parameters:
!     nbox = number of intervals into which the ranges of ph \in [0,pi], cos(th) \in[-1,1]
!       and ps \in [0,pi] are divided. The total number of boxes in the discretized Euler 
!       space is nbox**3.
!  Output: 
!     f_ol = ODF for olivine at the centers of each  box in the Euler space
!     f_opx = ODF for opx
!
   IMPLICIT NONE

   INTEGER :: iph,ith,ips,i,j,k,l,m,n,nbox,ia1,ia2,ia3,i1,i2,i3,nt,ntint,s_ol,ideg,ilimit,isteptype
   DOUBLE PRECISION :: pi,zero,one,tiny,cf
   DOUBLE PRECISION :: Gamma, rlam
!changed
   DOUBLE PRECISION :: tau,dtau,dtau2,taumax,taunew,taustart,e0dot
   DOUBLE PRECISION :: phdot,thdot,psdot,divpsdot
   DOUBLE PRECISION :: c1,drx,frac_opx
   DOUBLE PRECISION :: rmax,rn,vr,vrmax
   DOUBLE PRECISION :: cph,sph,phdot_ext,thdot_ext,psdot_ext,psdot_int
   DOUBLE PRECISION :: ph,th,ps,dph,dth,dcosth,dps,costh,dphdeg,dthdeg,dpsdeg
   DOUBLE PRECISION :: ph_fse,th_fse,ps_fse
   DOUBLE PRECISION :: phf,thf,psf
   DOUBLE PRECISION :: r12,r23,fmin,fmax,fintegral,lnf,lnf_nodrx,chideg,a11max,a12max,a13max
   DOUBLE PRECISION, DIMENSION(3) ::  edot,fse_axis,omdot,expect100
   DOUBLE PRECISION, DIMENSION(3,3) ::  a,o,g,goinv,vg,sr
!changed
   DOUBLE PRECISION, DIMENSION(0:90,0:90,0:90) :: pha,tha,psa,phm,thm,psm
   DOUBLE PRECISION, DIMENSION(0:90,0:90,0:90) :: ph_ol,th_ol,ps_ol        
   DOUBLE PRECISION, DIMENSION(0:90,0:90,0:90) :: ph_opx,th_opx,ps_opx        
!changed
   DOUBLE PRECISION, DIMENSION(0:90,0:90,0:90) :: f,f_nodrx,f_ol,f_opx
   CHARACTER(LEN=60):: fileout_ol, fileout_opx

   pi = 4.0d0*datan(1.0d0)
   zero = 0.0d0
   one = 1.0d0
   tiny = 1.0d-6

   OPEN (10, file='odftex_2orderstep.inp')
      DO i = 1, 3
        READ(10,*) (vg(i,j),j=1,3)  ! velocity gradient tensor (assumed constant)
      ENDDO 
      READ(10,*) s_ol  ! index of active olivine slip system
      READ(10,*) frac_opx  ! volume fraction of opx
! you will need to redimensionalize rlam by multiplying it by
! the strain rate scale Sqrt[E_{ij} E_{ij}/2] where E_{ij} is the dimensional
! strain rate tensor
      READ(10,*) rlam  ! dim'less recrystallization rate
      READ(10,*) taumax, ntint  ! maximum dim'less time and number of timesteps
! the option ilimit = 1 is only present to indicate how the texture is limited by 
! rotating it passively. Look for the character string "ilimit" in subroutine
! step. In practice, a limitation criterion based on the finite
! strain ellipsoid has to be added
      READ(10,*) ilimit  ! = 1 to limit texture based on FSE; 0 otherwise
      READ(10,*) nbox
      READ(10,*) ideg
      READ(10,*) fileout_ol
      READ(10,*) fileout_opx
   CLOSE (10)

! ideg determines whether output Eulerian angles are in degrees or radians
   if (ideg.eq.1) then
     cf = 180.0d0/pi
   else
     cf = one
   endif

! Strainrate tensor
   DO i = 1, 3; DO j = 1, 3
     sr(i,j) = 0.5*(vg(j,i) + vg(i,j))
   ENDDO; ENDDO

! Rescale vg and sr so that Sqrt[sr::sr/2] = 1
   e0dot = 0.0d0
   DO i = 1, 3; DO j = 1, 3
     e0dot = e0dot + sr(i,j)**2/2
   ENDDO; ENDDO
   e0dot = dsqrt(e0dot)

   DO i = 1, 3; DO j = 1, 3
     vg(i,j) = vg(i,j)/e0dot
     sr(i,j) = sr(i,j)/e0dot
   ENDDO; ENDDO

! Extrinsic spin (one-half the vorticity)
   omdot(1) = 0.5*(vg(3,2) - vg(2,3))
   omdot(2) = 0.5*(vg(1,3) - vg(3,1))
   omdot(3) = 0.5*(vg(2,1) - vg(1,2))

   dph = pi/nbox
   dcosth = 2.0d0/nbox
   dps = pi/nbox
   dtau = taumax/ntint
   dtau2 = dtau/2
   c1 = - 2.d0*(sr(1,1)**2 + sr(1,2)**2 + sr(1,1)*sr(2,2) + sr(2,2)**2 + sr(2,3)**2 + sr(3,1)**2)/5.d0

!  Initial uniform grid in the Euler space; grid points at centers of boxes
!  to avoid singularities
   DO i = 1, nbox; DO j = 1, nbox; DO k = 1, nbox
     pha(i,j,k) = (i - 0.5)*dph
     costh = -1.0 + (j - 0.5)*dcosth
     tha(i,j,k) = dacos(costh)
     psa(i,j,k) = (k - 0.5)*dps
     f(i,j,k) = 1.d0
     f_nodrx(i,j,k) = 1.d0
     CALL EULER_TO_DIRCOS(pha(i,j,k),tha(i,j,k),psa(i,j,k),a)
   ENDDO; ENDDO; ENDDO

   DO nt = 1, ntint  ! start of time-stepping loop

   taustart = (nt - 1)*dtau

! Half step to midpoint tau + dtau/2
   CALL STEP(1,pha,tha,psa,pha,tha,psa,phm,thm,psm,f,f_nodrx,nbox,omdot,sr,rlam,c1,ilimit,dtau2)

! Full step to tau + dtau
   CALL STEP(2,pha,tha,psa,phm,thm,psm,pha,tha,psa,f,f_nodrx,nbox,omdot,sr,rlam,c1,ilimit,dtau)

!  Normalize integral of ODF to unity
   CALL INTEGRAL_DRX(f,f_nodrx,nbox,fintegral)
!  write(6,*) 'integral =', fintegral
   DO i = 1, nbox; DO j = 1, nbox; DO k = 1, nbox
     f(i,j,k) = f(i,j,k)/fintegral
   ENDDO; ENDDO; ENDDO

! the subroutine EULERIAN_TRANSFORM is particularly fast if 
! the only active olivine slip system is s_ol = 1, because this slip
! system defines the reference Eulerian angles.
!  Transform Eulerian angles for ol and opx 
   CALL EULERIAN_TRANSFORM(nbox,s_ol,f,pha,tha,psa,f_ol,ph_ol,th_ol,ps_ol)
   if (frac_opx.gt.tiny) CALL EULERIAN_TRANSFORM(nbox,4,f,pha,tha,psa,f_opx,ph_opx,th_opx,ps_opx)

   taunew = taustart + dtau

   ENDDO  ! end of time-stepping loop

   OPEN (31,file=fileout_ol)
   OPEN (32,file=fileout_opx)
   DO i = 1, nbox; DO j = 1, nbox; DO k = 1, nbox
     write(31,1002) cf*ph_ol(i,j,k),cf*th_ol(i,j,k),cf*ps_ol(i,j,k),f_ol(i,j,k)
     if (frac_opx.gt.tiny) write(32,1002) cf*ph_opx(i,j,k),cf*th_opx(i,j,k),cf*ps_opx(i,j,k),f_opx(i,j,k)
   ENDDO; ENDDO; ENDDO
   CLOSE (31)
   CLOSE (32)

 1000 format(3i5,4(1pe13.5))
 1001 format(8(1pe14.6))
 1002 format(8(1pe12.4))

   STOP
   END PROGRAM ODFTEX_2ORDERSTEP

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   SUBROUTINE STEP(isteptype,ph1,th1,ps1,phm,thm,psm,ph2,th2,ps2,f,f_nodrx,nbox,omdot,sr,rlam,c1,ilimit,dt)
!  Advances solution by a time dt
!  (ph1,th1,ps1): initial Eulerian angles
!  (phm,thm,psm): Eulerian angles for which RHS is  calculated
!  (ph2,th2,ps2): final Eulerian angles
!  isteptype = 1: Euler half-step by dtau/2 to advance Eulerian angles along characteristics
!  isteptype = 2: full midpoint time step by dtau; also advances the ODF
   IMPLICIT NONE
   INTEGER :: i,j,k,nbox,ilimit,isteptype
   DOUBLE PRECISION :: rlam,c1,dt,cph,sph,phdot_ext,thdot_ext,psdot_ext,drx,psdot_int,phdot, &
    thdot,psdot,divpsdot,one,zero,lnf,lnf_nodrx
   DOUBLE PRECISION, DIMENSION(0:90,0:90,0:90) :: ph1,th1,ps1,ph2,th2,ps2,phm,thm,psm,f,f_nodrx        
   DOUBLE PRECISION, DIMENSION(3,3) :: sr
   DOUBLE PRECISION, DIMENSION(3) :: omdot

   zero = 0.0d0
   one = 1.0d0

   DO i = 1, nbox; DO j = 1, nbox; DO k = 1, nbox
     cph = dcos(phm(i,j,k)) 
     sph = dsin(phm(i,j,k))
     phdot_ext = omdot(3) + (omdot(2)*cph - omdot(1)*sph)/dtan(thm(i,j,k))
     thdot_ext = omdot(1)*cph + omdot(2)*sph
     psdot_ext = (omdot(1)*sph - omdot(2)*cph)/dsin(thm(i,j,k))
     if (ilimit.ne.1) CALL PSDOT_CALC_GENERAL(phm(i,j,k),thm(i,j,k),psm(i,j,k),sr,psdot_int,divpsdot)
     drx = rlam*(c1 + 2.d0*psdot_int**2)
     phdot = phdot_ext
     thdot = thdot_ext
     psdot = psdot_ext + psdot_int
     if (ilimit.eq.1) then
       drx = zero
       psdot_int = zero
       divpsdot = zero
       phdot = phdot_ext
       thdot = thdot_ext
       psdot = psdot_ext
     endif
     ph2(i,j,k) = ph1(i,j,k) + dt*phdot
     th2(i,j,k) = th1(i,j,k) + dt*thdot
     ps2(i,j,k) = ps1(i,j,k) + dt*psdot
     IF (isteptype.eq.2) then
       f(i,j,k) = f(i,j,k)*dexp(dt*(drx - divpsdot))
       f_nodrx(i,j,k) = f_nodrx(i,j,k)*dexp(-dt*divpsdot)
     ENDIF
   ENDDO; ENDDO; ENDDO

   END SUBROUTINE STEP

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   SUBROUTINE EULERIAN_TRANSFORM(nbox,s,f1,ph1,th1,ps1,f2,ph2,th2,ps2)
   IMPLICIT NONE
   DOUBLE PRECISION, DIMENSION(0:90,0:90,0:90) :: f1,ph1,th1,ps1        
   DOUBLE PRECISION, DIMENSION(0:90,0:90,0:90) :: f2,ph2,th2,ps2        
   DOUBLE PRECISION, DIMENSION(3,3) :: bmat,amat,bamat
   DOUBLE PRECISION :: zero,one,pi
   INTEGER :: s,nbox,i,j,k,l,m
   zero = 0.0d0
   one = 1.0d0
   pi = 4.0d0*datan(one)

   IF (s.eq.1) THEN
     DO i = 1, nbox; DO j = 1, nbox; DO k = 1, nbox
       ph2(i,j,k) = ph1(i,j,k)
       th2(i,j,k) = th1(i,j,k)
       ps2(i,j,k) = ps1(i,j,k)
       f2(i,j,k) = f1(i,j,k)
     ENDDO; ENDDO; ENDDO
     RETURN   
   ENDIF

   DO i=1,3; DO j=1,3
     bmat(i,j) = zero 
   ENDDO; ENDDO
   IF (s.eq.2) THEN
! Correct 20/03/25
     bmat(1,1) = one
     bmat(2,3) = one
     bmat(3,2) = -one
   ENDIF
   IF (s.eq.3) THEN
! Correct 20/03/25
     bmat(1,3) = one    
     bmat(2,2) = -one   
     bmat(3,1) = one   
   ENDIF
   IF (s.eq.4) THEN
! Correct 19/03/25
     bmat(1,2) = one
     bmat(2,3) = one
     bmat(3,1) = one
   ENDIF

   DO i = 1, nbox; DO j = 1, nbox; DO k = 1, nbox
     CALL EULER_TO_DIRCOS(ph1(i,j,k),th1(i,j,k),ps1(i,j,k),amat)
     bamat = MATMUL(bmat(:,:),amat(:,:)) 
     CALL DIRCOS_TO_EULER(bamat,ph2(i,j,k),th2(i,j,k),ps2(i,j,k))
! The values of rotated and unrotated ODFs are the same, just 
! assigned to different Eulerian angles
     f2(i,j,k) = f1(i,j,k) 
   ENDDO; ENDDO; ENDDO

   RETURN
   END SUBROUTINE EULERIAN_TRANSFORM

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   SUBROUTINE PSDOT_CALC_GENERAL(ph,th,ps,sr,psdot,divpsdot)
! Calculates intrinsic spin gdot and its derivatives
   IMPLICIT NONE
   DOUBLE PRECISION :: ph,th,ps,psdot,divpsdot
   DOUBLE PRECISION :: cph,sph,cth,sth,cps,sps
   DOUBLE PRECISION, DIMENSION (3,3) :: sr,a,daps
   
   CALL EULER_TO_DIRCOS(ph,th,ps,a)

   cph = dcos(ph)
   sph = dsin(ph)
   cth = dcos(th)
   sth = dsin(th)
   cps = dcos(ps)
   sps = dsin(ps)
! derivatives of a(i,j) w/r to ps
   daps(1,1) = -(cps*cth*sph) - cph*sps 
   daps(1,2) = cph*cps*cth - sph*sps
   daps(1,3) = cps*sth
   daps(2,1) = -(cph*cps) + cth*sph*sps
   daps(2,2) = -(cps*sph) - cph*cth*sps
   daps(2,3) = -(sps*sth)
   daps(3,1) = 0.d0
   daps(3,2) = 0.d0
   daps(3,3) = 0.d0

   psdot = a(1,2)*(a(2,1)*sr(1,2) + a(2,2)*sr(2,2) + a(2,3)*sr(2,3))  &
         + a(1,1)*(a(2,1)*sr(1,1) + a(2,2)*sr(1,2) + a(2,3)*sr(3,1))  &
         + a(1,3)*(a(2,2)*sr(2,3) + a(2,1)*sr(3,1) + a(2,3)*sr(3,3)) 
   divpsdot = daps(1,2)*(a(2,1)*sr(1,2) + a(2,2)*sr(2,2) + a(2,3)*sr(2,3)) +   &
         a(1,2)*(daps(2,1)*sr(1,2) + daps(2,2)*sr(2,2) + daps(2,3)*sr(2,3)) +      &
         daps(1,1)*(a(2,1)*sr(1,1) + a(2,2)*sr(1,2) + a(2,3)*sr(3,1)) +        &
         a(1,1)*(daps(2,1)*sr(1,1) + daps(2,2)*sr(1,2) + daps(2,3)*sr(3,1)) +      &
         daps(1,3)*(a(2,2)*sr(2,3) + a(2,1)*sr(3,1) + a(2,3)*sr(3,3)) +        &
         a(1,3)*(daps(2,2)*sr(2,3) + daps(2,1)*sr(3,1) + daps(2,3)*sr(3,3))

   RETURN
   END SUBROUTINE PSDOT_CALC_GENERAL

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   SUBROUTINE PSDOT_CALC(ph,th,ps,e1,e2,psdot,divpsdot)
! Calculates psdot and its derivative divpsdot w/r to ps
   IMPLICIT NONE
   DOUBLE PRECISION :: ph,th,ps,e1,e2,e3,psdot,divpsdot 
   DOUBLE PRECISION :: cph,sph,c2ph,s2ph
   DOUBLE PRECISION :: cth,sth,c2th,s2th
   DOUBLE PRECISION :: c2ps,s2ps
   DOUBLE PRECISION :: e1m2,e2m3
   DOUBLE PRECISION :: bigf,bigg,bigh,bigp,bigq

   e3 = - e1 - e2
   cph = dcos(ph)
   sph = dsin(ph)
   c2ph = dcos(2*ph)
   s2ph = dsin(2*ph)
   cth = dcos(th)
   sth = dsin(th)
   c2th = dcos(2*th)
   s2th = dsin(2*th)
   c2ps = dcos(2*ps)
   s2ps = dsin(2*ps)
   e1m2 = e1 - e2
   e2m3 = e2 - e3
   bigf = - s2ph*cth
   bigg = sph**2*cth**2 - cph**2
   bigh = - sth**2
   bigp = e1m2*bigf
   bigq = e1m2*bigg + e2m3*bigh

   psdot = 0.5*(bigp*c2ps + bigq*s2ps)
   divpsdot = -bigp*s2ps + bigq*c2ps

   RETURN
   END SUBROUTINE PSDOT_CALC

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   SUBROUTINE EULER_TO_DIRCOS(ph,th,ps,o)
!  Calculate direction cosines corresponding to given Eulerian angles

   IMPLICIT NONE

   DOUBLE PRECISION :: ph,th,ps
   DOUBLE PRECISION :: cph,cth,cps,sph,sth,sps
   DOUBLE PRECISION, DIMENSION(3,3) :: o

   cph = dcos(ph)
   cth = dcos(th)
   cps = dcos(ps)
   sph = dsin(ph)
   sth = dsin(th)
   sps = dsin(ps)

   o(1,1) =   cps*cph  - cth*sph*sps 
   o(1,2) =   cps*sph  + cth*cph*sps
   o(1,3) =   sps*sth 
   o(2,1) = - sps*cph  - cth*sph*cps 
   o(2,2) = - sps*sph  + cth*cph*cps 
   o(2,3) =   cps*sth 
   o(3,1) =   sph*sth 
   o(3,2) = - cph*sth 
   o(3,3) =   cth 

   return

   END SUBROUTINE EULER_TO_DIRCOS

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   SUBROUTINE DIRCOS_TO_EULER(o,ph,th,ps)
!  Calculates Eulerian angles corresponding to a given matrix of direction cosines

   IMPLICIT NONE

   DOUBLE PRECISION :: ph,th,ps,pi
   DOUBLE PRECISION, DIMENSION(3,3) :: o

   pi = 4.0d0*datan(1.0d0)

   th = dacos(o(3,3))
   ph = datan2(o(3,1),-o(3,2))
   ps = datan2(o(1,3),o(2,3))
 
   return

   END SUBROUTINE DIRCOS_TO_EULER

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   SUBROUTINE INTEGRAL_DRX(odf,odf_nodrx,nbox,integral)
!  Approximate calculation of (1/(2 Pi**2)) Integral(f dg)
!  Uses dg/odf_nodrx as an estimate of the elemental volume of the Euler
!    space for point (i,j,k)
!  Result = 1 exactly if drx is turned off

   IMPLICIT NONE

   INTEGER :: i,j,k,nbox,iphm1,ithm1,ipsm1
   DOUBLE PRECISION :: pi,integral,df,ph_side,x_side,ps_side,xp,xm,box
   DOUBLE PRECISION :: dcosth,dph,dps,dg
   DOUBLE PRECISION, DIMENSION(0:90,0:90,0:90) :: odf,odf_nodrx

   pi = 4.0d0*datan(1.0d0)

   dcosth = 2.0d0/nbox
   dph = pi/nbox
   dps = pi/nbox
   dg = dcosth*dph*dps

   integral = 0.0d0
   DO i = 1, nbox; DO j = 1, nbox; DO k = 1, nbox
   integral = integral + dg*odf(i,j,k)/odf_nodrx(i,j,k)
   ENDDO; ENDDO; ENDDO
   integral = integral/(2.0*pi**2)

   END SUBROUTINE INTEGRAL_DRX
