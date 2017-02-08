
clean:
	rm -r *.aux *.pdf *~ *.log *.nav *.out *.snm _minted* *.toc

01: 01a.pdf 01b.pdf 01c.pdf 01d.pdf 01e.pdf 01f.pdf

02: 02a.pdf 02b.pdf 02c.pdf 02d.pdf

03: 03a.pdf 03b.pdf 03c.pdf 03d.pdf

04: 04a.pdf 04b.pdf

%.pdf: %.tex
	pdflatex -shell-escape $*
	cp $@ ../../jbg-web/teaching/CSCI_7000
