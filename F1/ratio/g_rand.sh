for i in {1..30};do split -l 21 --additional-suffix .m$i --numeric-suffixes ../cv30/m$i.id;done set1

mkdir set1 set9 test

mv x00.m1 set1
mv x00.m2 set1
mv x00.m3 set1
mv x01.m4 set1
mv x01.m5 set1
mv x01.m6 set1
mv x02.m7 set1
mv x02.m8 set1
mv x02.m9 set1
mv x03.m10 set1
mv x03.m11 set1
mv x03.m12 set1
mv x04.m13 set1
mv x04.m14 set1
mv x04.m15 set1
mv x05.m16 set1
mv x05.m17 set1
mv x05.m18 set1
mv x06.m19 set1
mv x06.m20 set1
mv x06.m21 set1
mv x07.m22 set1
mv x07.m23 set1
mv x07.m24 set1
mv x08.m25 set1
mv x08.m26 set1
mv x08.m27 set1
mv x09.m28 set1
mv x09.m29 set1
mv x09.m30 set1

mv x* set9

cat set1/* >ratio1.id
cat set9/* >ratio9.id

perl g_randtestlist.pl ratio9.id test
perl g_randtraintest.pl ../cv30/seqname.id one2nine

