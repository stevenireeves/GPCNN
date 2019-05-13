% Function to do GP interpolation in 2D for various lratio

function Zinterp = GPinterp(Zcc, lx, ly, Kinv, ks)
  Zinterp = zeros(lx*size(Zcc)(1), ly*size(Zcc)(2)); 
  one = ones(5,1); 
  fc = one'*Kinv/(one'*Kinv*one); 
  for i = 2:size(Zcc)(1)-1
    for j = 2:size(Zcc)(2)-1
    stencil = [Zcc(i,j+1); Zcc(i-1, j); Zcc(i,j); Zcc(i+1,j); Zcc(i,j-1)];
    fbar = fc*stencil;
    aug = Kinv*(stencil - fbar); 
      for iref = 1:lx
        for jref = 1:ly
          iter = iref + lx*(jref-1);
          ii = (i-1)*lx + iref;
          jj = (j-1)*ly + jref;
          Zinterp(ii,jj) = round(fbar +  ks(:,iter)'*aug);
        end
      end
    end
  end
end