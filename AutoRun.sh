git clone git@github.com:pfittipaldi/prova
cd prova
cd 16\ 02\ QSFK\ Sweep
python ParSweep.py
git add *
git commit -m "Last Run: $(date)"
git push
rm -rf prova
