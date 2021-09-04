rv="r9v1 r7v1 r5v1 r3v1 r1v1 r1v3 r1v5 r1v7 r1v9"
model="gb lgb"
trait="7 8 9"

dir="/data/YanJ/PROJECT/cropGBM/"

mkdir -p output

o='output'
for t in $trait;do
for vs in $rv;do
for x in {1..30};do
for m in $model;do
	python modelTest.py --training $dir/data/G_6210_32559_snp.data --trainlist $dir/fig1/ratio/ratio1.id --testlist $dir/fig1/ratio/test/$vs.m$x.id --n $t --snpinfo $dir/fig1/cv30/T$t.snp.snpinfo --output $o/$vs.T$t.$m.m$x --assess no --mask yes --method $m >$o/$vs.T$t.$m.m$x.cor
done
done
done
done

m="rbsaf"
for t in $trait;do
for vs in $rv;do
for x in {1..30};do
	python modelTest.py --training $dir/data/G_6210_32559_snp.data --trainlist $dir/fig1/ratio/ratio1.id --testlist $dir/fig1/ratio/test/$vs.m$x.id --n $t --snpinfo $dir/fig1/cv30/snp.snpinfo --output $o/$vs.T$t.$m.m$x --assess no --mask yes --method $m --fixidx $dir/fig1/cv30/T$t.fixidx >$o/$vs.T$t.$m.m$x.cor
done
done
done
