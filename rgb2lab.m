function [lvec,avec,bvec] = rgb2lab(img)
img = double(img);
img = img./255.0;
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);
[height,width] = size(R);
for i=1:height
    for j=1:width
        if(R(i,j) <= 0.04045)
            r = R(i,j)/12.92;
        else
            r = power((R(i,j)+0.055)/1.055,2.4);
        end
        if(G(i,j) <= 0.04045)
            g = G(i,j)/12.92;
        else
            g = power((G(i,j)+0.055)/1.055,2.4);
        end
        if(B(i,j) <= 0.04045)
            b = B(i,j)/12.92;
        else
            b = power((B(i,j)+0.055)/1.055,2.4);
        end
        X = r*0.4124564 + g*0.3575761 + b*0.1804375;
		Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
		Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
        epsilon = 0.008856;	
		kappa   = 903.3;		

		Xr = 0.950456;	
		Yr = 1.0;		
		Zr = 1.088754;	

		xr = X/Xr;
		yr = Y/Yr;
		zr = Z/Zr;
		if(xr > epsilon)	
            fx = power(xr, 1.0/3.0);
        else
            fx = (kappa*xr + 16.0)/116.0;
        end
		if(yr > epsilon)	
            fy = power(yr, 1.0/3.0);
        else
            fy = (kappa*yr + 16.0)/116.0;
        end
		if(zr > epsilon)	
            fz = power(zr, 1.0/3.0);
        else
            fz = (kappa*zr + 16.0)/116.0;
        end
        lvec(i,j) = 116.0*fy-16.0;
		avec(i,j) = 500.0*(fx-fy);
		bvec(i,j) = 200.0*(fy-fz);
    end
end
end
