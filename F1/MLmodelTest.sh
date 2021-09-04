model="knn svr rf gb mlp"
trait="7 8 9"
feature="snp"
dir="/data/YanJ/PROJECT/cropGBM/"

mkdir -p output

o='output'
for t in $trait;do
for f in $feature;do
for x in {1..30};do
for m in $model;do
	python modelTest.py --training $dir/data/G_6210_32559_$f.data --testlist m$x.id --n $t --snpinfo T$t.$f.snpinfo --output $o/$f.T$t.$m.m$x --assess no --mask yes --method $m >$o/$f.T$t.$m.m$x.cor
done
done
done
done

m="rbsaf"
for t in $trait;do
for f in $feature;do
for x in {1..30};do
for m in $model;do
	python modelTest.py --training $dir/data/G_6210_32559_snp.data --testlist m$x.id --n $t --snpinfo $dir/fig1/cv30/snp.snpinfo --output $o/$f.T$t.$m.m$x --assess no --mask yes --method $m --fixidx $dir/fig1/cv30/T$t.fixidx >$o/$f.T$t.$m.m$x.cor
done
done
done
done

