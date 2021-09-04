tx="m1 m2 m3 m4 m5"

dir="/data/YanJ/PROJECT/cropGBM/"

mkdir -p output

o='output'

model="rbsnf"
trait="1 2 3 4 5 6"
for x in $tx;do
for t in $trait;do
for m in $model;do
	python modelTest.py --training $dir/data/1428.gca.data --testlist $x.id --n $t --snpinfo $dir/fig1/cv30/snp.snpinfo --output $o/T$t.$m.$x --assess no --mask yes --method $m >$o/T$t.$m.$x.log
done
done
done

model="lgb"
trait="1 2 3"
for x in $tx;do
for t in $trait;do
for m in $model;do
	python modelTest.py --training $dir/data/1428.gca.data --testlist $x.id --n $t --snpinfo $dir/fig1/cv30/snp.snpinfo --output $o/T$t.$m.$x --assess no --mask yes --method $m --type cla --evaluate 3 >$o/T$t.$m.$x.log
done
done
done

trait="4 5 6"
for x in $tx;do
for t in $trait;do
for m in $model;do
	python modelTest.py --training $dir/data/1428.gca.data --testlist $x.id --n $t --snpinfo $dir/fig1/cv30/snp.snpinfo --output $o/T$t.$m.$x --assess no --mask yes --method $m --type cla --evaluate 2 >$o/T$t.$m.$x.log
done
done
done
