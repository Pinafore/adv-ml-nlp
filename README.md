# adv-ml-nlp

Student Presentations
=

Overview
--------

This is a graduate seminar.  The good news is that we're not going to
assign onerous homeworks or have soul-crushing final exams.  However,
because it's a graduate seminar, you're going to have to do work to
make sure that you understand the material.

I believe that one of the best ways to understand material is to teach
it to someone else.  This is why I'm asking students to present
material once a week to the course.

Moreover, one of the goals of a PhD program is to help students
communicate clearly.  Classroom presentations are a good way of
getting this practice.

Outline
-------

You'll propose what you're going to talk about two weeks before your
presentation and get feedback in person from the class.

You need to be able to articulate what you're going to present.  Don't
just say, "I'm going to do a presentation".  You need to articulate
what material you're going to present.  Describe a particular model,
algorithm, or problem that you'll present.

For example (for a mathematical treatment): 

1.  I'll present the derivation of variational inference for a Dirichlet Process Mixture Model with a multinomial base distribution.
1.  I assume that people remember variational inference in general, but I'll need to review the stick breaking process.
1.  We'll talk about truncating the stick, and I won't talk about hybrid methods.

You should be able to answer questions about the content you'll
present.  E.g., what you'll assume, how it will be different from the
previous week's content, etc.  While I have suggestions for what to
present, I think the course works better when people can think deeply
about content and come up with their own ideas.

Revised Outline
---------------

You'll then give an expanded outline on Piazza to reflect the feedback
you got in class.  If you weren't prepared in class, then you can do
private followups with the instructors if you have questions/concerns.
(Remember to include your github ID so we can add you to the
repository to submit your material.)

Things to Think About for Mathematical Treatment
------------------------------------------------

For the mathematical treatment, don't just manipulate symbols.  You want to:
1.  Give a high-level picture of why we're doing what we're doing
1.  For any manipulation explain what the step is meant to accomplish and what tricks you're using (e.g., using highlighting)
1.  When you've finished a coherent set of steps, give a big-picture recap of what happened and why

Things to Think About for Hands-on Demonstration
------------------------------------------------

The demonstration should *not* be a presentation; it should be
interactive.  Just as I did in the first classes, it should allow
students to practice their understanding of the material.  Thus it
must be interactive.

You should have clear steps for students to work through in groups.
Break up the larger problem into steps for them to solve.  Explicitly
put the problem on the chalkboard or on the projector along with all
of the informaiton they need to solve the problem.

Know when you're going to pause for students to work through the
problem on their own.  Then present the solution *step-by-step* so
that if a student got the wrong answer they can see why they got it
wrong.

Runthrough
---------------

You'll go over your presentation with the course assistant the week
before you present.  They will look for the following:
1.  *Correctness*: Are you presenting the content accurately?
1.  *Completeness*: Are you presenting all of the information for students need?
1.  *Usefullness*: Is the information you're presenting informative and useful for students?

Final Submission
----------------

If you generate slides, please use either Beamer or markdown.  Include
a Makefile that by default generates everything necessary (it's fine
to generate material from a Python/Ruby/Julia/Perl script, just make
sure that's part of the Makefile).

Create a new directory for either your presentation/mathematical
treatment and include all material needed for your Makefile to
generate your results.
