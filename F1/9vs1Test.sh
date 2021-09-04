rv="r1 r2 r3 r4 r5 r6 r7 r8 r9"
model="gb lgb"
trait="7 8 9"

dir="/data/YanJ/PROJECT/cropGBM/"

mkdir -p output

o='output'
for t in $trait;do
for vs in $rv;do
for x in {1..30};do
for m in $model;do
	python modelTest.py --training $dir/data/G_6210_32559_snp.data --trainlist $dir/fig1/ratio/one2nine/$vs.m$x.nine --testlist $dir/fig1/ratio/one2nine/$vs.m$x.one --n $t --snpinfo $dir/fig1/cv30/T$t.snp.snpinfo --output $o/$vs.T$t.$m.m$x --assess no --mask yes --method $m >$o/$vs.T$t.$m.m$x.cor
done
done
done
done

m="rbsaf"
for t in $trait;do
for vs in $rv;do
for x in {1..30};do
	python modelTest.py --training $dir/data/G_6210_32559_snp.data --trainlist $dir/fig1/ratio/one2nine/$vs.m$x.nine --testlist $dir/fig1/ratio/one2nine/$vs.m$x.one --n $t --snpinfo $dir/fig1/cv30/snp.snpinfo --output $o/$vs.T$t.$m.m$x --assess no --mask yes --method $m --fixidx $dir/fig1/cv30/T$t.fixidx >$o/$vs.T$t.$m.m$x.cor
done
done
done


