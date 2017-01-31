import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(pred_func, X, y, filename):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = np.array(map(pred_func, np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.savefig(filename)


class Model:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.dim = len(w1)

    def model_and_data_slide(self, x, y):
        res = "\\begin{frame}{Data and Model}\n"

        res += "\\begin{columns}\n"
        res += "\\column{.5\\linewidth}\n"
        res += "\\begin{block}{Data}\n"

        res += "\\begin{tabular}{ccc}\n"
        res += "$x_1$ & $x_2$ & $y$ \\\\ \n \\hline \n"
        for xx, yy in zip(x, y):
            res += "%0.2f & %0.2f & %0.2f \\\\ \n" % (xx[0], xx[1], yy)
        res += "\\end{tabular}\n"
        res += "\\end{block}\n"

        res += "\\column{.5\\linewidth}\n"

        res += "\\begin{block}{First Layer}\n"
        res += "\\begin{equation}\n"
        res += "w^{(1)} = \\begin{bmatrix}\n"
        res += "%0.2f & %0.2f \\\\ \n" % (self.w1[0][0], self.w1[0][1])
        res += "%0.2f & %0.2f \\\\ \n" % (self.w1[1][0], self.w1[1][1])
        res += "\\end{bmatrix}\n"
        res += "\\end{equation}\n"

        res += "\\begin{equation}\n"
        res += "b^{(1)} = \\begin{bmatrix}\n"
        res += "%0.2f & %0.2f \\\\ \n" % (self.b1[0][0], self.b1[0][1])
        res += "\\end{bmatrix}\n"
        res += "\\end{equation}\n"

        res += "\\end{block}\n"

        res += "\\begin{block}{Second Layer}\n"
        res += "\\begin{equation}\n"
        res += "w^{(2)} = \\begin{bmatrix}\n"
        res += "%0.2f & %0.2f \\\\ \n" % (self.w2[0][0], self.w2[0][1])
        res += "\\end{bmatrix}\n"
        res += "\\end{equation}\n"

        res += "\\begin{equation}\n"
        res += "b^{(2)} = %0.2f \n" % (self.b2)
        res += "\\end{equation}\n"

        res += "\\end{block}\n"

        res += "\\end{columns}\n"

        res += "Using ReLU as non-linearity"

        res += "\\end{frame}\n"

        return res

    @staticmethod
    def act(observation):
        """
        The activation function we're using
        """

        return np.array([0.0 if x < 0.0 else x for x in observation])

    @staticmethod
    def dact(observation):
        """
        Derivative of the activation function
        """

        return np.array(0.0 if x < 0.0 else 1.0 for x in observation)

    def output_layer(self, hidden):
        z2 = hidden.dot(self.w2[0]) + self.b2
        a2 = z2 if z2 > 0.0 else 0.0
        return a2

    @staticmethod
    def softmax(vals):
        res = np.exp(vals)
        return res / np.sum(res)

    def hidden_layer(self, x):
        z1 = x.dot(self.w1) + self.b1
        a1 = np.array(map(self.act, z1))[0]

        return a1

    def hidden_computation(self, val, idx):
        res = ""
        for ii in range(self.dim):
            res += "\\begin{align}\n\t"
            res += "a^{(1)}_{%i, %i} & = " % (idx, ii)
            res += "f("

            for jj, vv in enumerate(val):
                res += "w^{(1)}_{%i, %i} \cdot %0.2f + " % (ii, jj, vv)
            res += "b_{%i}) \\\\ \n" % ii

            res += "\t& = f("
            for jj, vv in enumerate(val):
                res += "%0.2f \cdot %0.2f + " % (self.w1[ii, jj], vv)
            res += "%0.2f) \n" % self.b1[0][ii]
            res += "\\end{align}\n"
            res += "\\pause\n"

        return res

    def plot_hidden_space(self, x):
        return None

    def output_computation(self, hidden, idx):
        res = ""
        for ii in [0]:
            res += "\\begin{align}\n\t"
            res += "a^{(3)}_{%i, %i} & = " % (idx, ii)
            res += "f("

            for jj, vv in enumerate(hidden):
                res += "w^{(2)}_{%i, %i} \cdot %0.2f + " % (ii, jj, vv)
            res += "b_{%i}) \\\\ \\pause\n" % ii

            res += "\t& = f("
            for jj, vv in enumerate(hidden):
                res += "%0.2f \cdot %0.2f + " % (self.w2[ii, jj], vv)
            res += "%0.2f) \n" % self.b2
            res += "\\end{align}\n"
            res += "\\pause\n"

        return res

    def predict(self, observation, skip_hidden=False):
        if not skip_hidden:
            hidden = self.hidden_layer(observation)
        else:
            hidden = observation

        return 1 if self.output_layer(hidden) > 0.5 else 0

    def computation_slide(self, observation, idx, correct):
        """
        Walks through the activation of model
        """

        res = ""
        res += "\\begin{frame}{Prediction for $x_%i=(%0.2f, %0.2f)$}\n" % \
                                                     (idx, observation[0],
                                                      observation[1])

        res += "\n\\pause\n"
        res += "\\begin{itemize}\n"
        res += "\\item Hidden Computation\n "
        res += self.hidden_computation(observation, idx)
        hidden = self.hidden_layer(observation)
        res += "\\item Hidden Layer: %s\n" % str(hidden)
        res += "\\item Output Answer\n"
        res += str(self.output_computation(hidden, idx))
        score = self.output_layer(hidden)
        res += "\item Prediction: %0.2f, Error: %0.2f" % \
               (score, (correct - score) ** 2)

        res += "\\end{itemize}\n"
        res += "\\end{frame}\n"

        return res


if __name__ == "__main__":
    x, y = (np.array([[1, 1], [1, 0], [0, 0], [0, 1]]),
            np.array([0, 1, 0, 1]))
    model = Model(np.array([[1, 1], [1, 1]]),
                  np.array([[-1, 0]]),
                  np.array([[-2, 1]]),
                  0.0)

    print(model.model_and_data_slide(x, y))

    for ii, vv in enumerate(x):
        print(model.computation_slide(vv, ii, y[ii]))

    plt.title("Original Space")
    plot_decision_boundary(lambda z: model.predict(z), x, y,
                           "../lectures/deep/relu_ex_orig.png")

    h = np.array(map(model.hidden_layer, x))
    plt.title("Hidden Space")
    plot_decision_boundary(lambda z: model.predict(z, skip_hidden=True), h, y,
                           "../lectures/deep/relu_ex_embed.png")

    print("\\begin{frame}\n")
    print("\t\\only<1>{\gfx{relu_ex_orig}{.8}}")
    print("\t\\only<2>{\gfx{relu_ex_embed}{.8}}")
    print("\\end{frame}\n")
