function saliency = Saliency(img)
s = size(img);
s = s(1:2);
l = ones(s);
a = ones(s);
b = ones(s);
[l,a,b] = rgb2lab(img);
avgl = mean2(l);
avga = mean2(a);
avgb = mean2(b);
salmap = ones(s);
for i = 1:s(1)
    for j = 1:s(2)
        salmap(i,j) = (l(i,j)-avgl)*(l(i,j)-avgl) +(a(i,j)-avga)*(a(i,j)-avga) +(b(i,j)-avgb)*(b(i,j)-avgb);
    end
end
MAX = max(max(salmap));
MIN = min(min(salmap));
range = MAX - MIN;
salmap = salmap - MIN;
salmap = salmap./range;
saliency = salmap;
end

