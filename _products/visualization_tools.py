
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
pd.options.mode.use_inf_as_na = True
from _products.utility_fnc import *

def gaussian_plot(xarrays, mus, stds, priors=[1,1], verbose=False, title='Gaussian Plot', m_label=['a', 'b'],
                  showit=True, fignum=1, legend=True):
    for xarray, mu, std, prior in zip(xarrays, mus, stds, priors):
        Visualizer().basic_plot(xarray, generate_gaussian(xarray, mu, std, prior, verbose), xlabel='x', ylabel='prob',
               title=title, show=showit, fig_num=fignum, m_label=m_label, legend=legend)
    if showit:
        plt.show()

class Visualizer:

    """ a lot of visualization methods
        There are:
                 ploting methods:
                    * dict_bar_plotter(): uses a dict to make a bar plot
                    *
                 stdout put methods
                    * print_test_params: takes a dictionary of paramter names and values and prints them to stdout
    """

    def fancy_plot_bar(self, df, feats, figsize=(20, 20), title='', x_label='', y_label='',
                    width=.5, offy=.05, offx=.01):
        axb = df.plot.bar()
        axb.set_title(title)
        ymx, ymn = df.max(axis=1)[0], df.min(axis=1)[0]
        axb.set_ylim(ymn-offy*ymn, ymx+offx*ymx)
        return

    def make_me_a_box(self, box_data, use_cols, title='the box plot title', fontdict=None,
                      figsize=(20, 20), savefigure=False, figname='Box_Plot',
                      format='svg', mainfsize=12):
        font = {'size': mainfsize,
                'weight': 'bold'}
        matplotlib.rc('font', **font)
        if fontdict is None:
            fontdict = {
                # 'family': 'serif',
                'family': 'sans-serif',
                # 'family': 'monospace',
                'style': 'normal',
                'variant': 'normal',
                'weight': 'heavy',
                'size': '15',
            }
        fig, ax = plt.subplots(1, 1, figsize=figsize, )
        ax.boxplot(box_data, vert=False,
                   labels=use_cols, )
        # ax2.set_title('Top {} features for Permutation Importance '.format(-1 * lmt), x = -0.02)
        ax.set_title(title, fontdict=fontdict, )
        # plt.title(title, fontdict=fontdict)
        ax.set_yticklabels(use_cols, fontdict=fontdict)
        fig.tight_layout()
        if savefigure:
            plt.savefig(figname, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=format,
                        transparent=False, bbox_inches=None, pad_inches=0.1,
                        frameon=None, metadata=None)
        plt.show()


    def print_test_params(self, param_d):
        print('Test Parameters:')
        for p in param_d:
            print('                 * {0}{1}'.format(p, param_d[p]))
        return

    def dict_bar_plotter(self, bar_dict, xlabel='Number of Hidden Neurons', ylabel='Time to train seconds',
                         title='Time to Complete for different Hidden neurons', save_fig=False, fig_name='',
                         font_dict=None, width=.5, color='red'):
        y_pos = np.arange(len(bar_dict))
        bar_dict = sort_dict(bar_dict)
        performance = bar_dict.values()
        labels = list(bar_dict.keys())

        plt.barh(y_pos, performance, align='center', alpha=0.5, color=color,)
        plt.yticks(y_pos, labels)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if save_fig:
            plt.savefig(fig_name)
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        # classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        specificity = cm[0][0] / (cm[0][0] + cm[0][1])
        sensitivity = cm[1][1] / (cm[1][0] + cm[1][1])
        overall_acc = (cm[1][1] + cm[0][0]) / (cm[1][0] + cm[1][1] + cm[0][0] + cm[0][1])
        precision = (cm[0][0] / (cm[0][0] + cm[1][0]))
        print('Accuracy: {:.3f}'.format(overall_acc))
        print('Recall: {:.3f}'.format(sensitivity))
        print('Specificity: {:.3f}'.format(specificity))
        print('Precision: {:.3f}'.format(precision))
        title = 'Accuracy: {:.3f}\nrecall: {:.3f}\nprecision: {:.3f}\nspecificity: {:.3f}'.format(overall_acc,
                                                                                                  sensitivity,
                                                                                                  precision,
                                                                                                  specificity)
        rd = {'Accuracy':overall_acc, 'Sensitivity':sensitivity,
              'Precision':precision, 'Specificity':specificity, 'CM':cm}
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        rd['ax'] = ax
        return rd

    def basic_plot(self, x, y, xlabel='xlabel', ylabel='ylabel', title='K value vs accuracy',
                   marker='x', show=False, fig_num=None, m_label=[''], legend=False):
        # artis for this plot
        art = None
        if fig_num is None:
            plt.figure()
        elif fig_num == 'ignore':
            pass
        else:
            plt.figure(fig_num)
        art = plt.plot(x,y,marker)
        #plt.scatter(x,y,color=color, marker=marker,label=m_label)
        if legend:
            plt.legend([m_label])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if show:
            plt.show()
        return art[0]

    def basic_plot_scatter(self, x, y, color='r', xlabel='xlabel', ylabel='ylabel', title='K value vs accuracy',
                   marker='x', show=False, fig_num=None, m_label=''):
        if fig_num is None:
            plt.figure()
        elif fig_num == 'ignore':
            pass
        else:
            plt.figure(fig_num)
        #plt.plot(x,y,color=color, marker=marker,label=m_label)
        plt.scatter(x,y,color=color, marker=marker,label=m_label)
        lgd = plt.legend(loc='best')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if show:
            plt.show()

    def sub_plotter(self, xarray, yarray, xlabels, ylabels, titles, markers, sharex='none', sharey='none', show=False,
                    fig_num=None, orientation='v'):
        # set up the subplot arrays using the
        # length of xarray
        num_plots = len(xarray)
        if orientation == 'v':
            fig, axs = plt.subplots(nrows=num_plots, ncols=1, sharex=sharex, sharey=sharey)
        else:
            fig, axs = plt.subplots(nrows=1, ncols=num_plots, sharex=sharex, sharey=sharey)


        for i in range(num_plots):
            axs[i].plot(xarray[i], yarray[i])
            axs[i].set_xlabel(xlabel=xlabels[i])
            axs[i].set_ylabel(ylabel=ylabels[i])
            axs[i].set_title(titles[i])
        if show:
            plt.show()

    def multi_plot(self, xarray, yarray, xlabel='x label', ylabel='y label',
                            title='MULTIPLOT TITLE', fig_num=None, legend_array=['me','you'], marker_array=['x', 'x'], show=False,
                            show_last=False, save=False, fig_name='Fig'):
        found = False
        l = len(xarray)
        cnt = 0
        arts = list()
        for x, y, m, la in zip(xarray, yarray, marker_array, legend_array):
            if fig_num is None and not found:
                fig_num = plt.figure().number
            #print('Fig num',fig_num)
            if show_last:
                if cnt < l-1:
                    a = self.basic_plot(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, fig_num=fig_num,
                                        m_label=[la], marker=m, show=False)
                    arts.append(a)
                else:
                    a = self.basic_plot(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, fig_num=fig_num,
                                    m_label=legend_array, marker=m, show=True, legend=True)
                    arts.append(a)
                cnt += 1
            else:
                a = self.basic_plot(x=x, y=y, xlabel=xlabel, ylabel=ylabel, title=title, fig_num=fig_num,
                                m_label=[la], marker=m, show=False)
                arts.append(a)
        lgd = plt.legend(arts, legend_array, loc='best')
        if save:
            plt.savefig(fig_name)
        plt.show()
        return fig_num


    def multi_plot_scatter(self, xarray, yarray, color_array=['r', 'b'], xlabel='x label', ylabel='y label',
                            title='MULTIPLOT TITLE', fig_num=None, legend_array=['me','you'], marker_array=['x', 'x'], show=False,
                            show_last=False):
        found = False
        l = len(xarray)
        cnt = 0
        for x, y, c, la, m in zip(xarray, yarray, color_array, legend_array, marker_array):
            if fig_num is None and not found:
                fig_num = plt.figure().number
            #print('Fig num',fig_num)
            if show_last:
                if cnt < l-1:
                    self.basic_plot_scatter(x=x, y=y, color=c, xlabel=xlabel, ylabel=ylabel, title=title, fig_num=fig_num,
                                    m_label=la, marker=m, show=False)
                else:
                    self.basic_plot_scatter(x=x, y=y, color=c, xlabel=xlabel, ylabel=ylabel, title=title, fig_num=fig_num,
                                    m_label=la, marker=m, show=True)
                cnt += 1
            else:
                self.basic_plot_scatter(x=x, y=y, color=c, xlabel=xlabel, ylabel=ylabel, title=title, fig_num=fig_num,
                                m_label=la, marker=m, show=show)
        return fig_num


    def bi_class_colored_scatter(self, x, y, class_dict, fig_num=None, legend=['class 0', 'class 1'], annotate=False, show=True,
                                 xl='x', yl='y', title='title'):
        for X, Y in zip(x,y):
            plt.scatter(X[0], X[1],  c=class_dict[Y])
        plt.title(title)
        plt.xlabel(xl)
        plt.ylabel(yl)
        leg = plt.legend(legend, loc='best', borderpad=0.3, shadow=False, markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        if show:
            plt.show()


    def bi_class_scatter3D(self, x, y, class_dict, fig_num=None, legend=['class 0', 'class 1'], annotate=False, show=True, treD=False,
                           xl = 'x', yl='y', zl='z', cols=(0, 1, 2), title='3D Class Scatter'):

        a = cols[0]
        b = cols[1]
        c = cols[2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for X, Y in zip(x, y):
            ax.scatter(X[a], X[b], X[c], c=class_dict[Y])
        plt.legend(['non adopters', 'adopters'])
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_zlabel(zl)
        plt.title(title)
        plt.show()

    def fancy_scatter_plot(self, x, y, styl, title, c, xlabel, ylabel, labels, legend,
                           annotate=True, s=.5, show=False):

        for z1, z2, label in zip(x, y, labels):
            plt.scatter(z1, z2, s=s, c=c)
            if annotate:
                plt.annotate(label, (z1, z2))

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        leg = plt.legend([legend], loc='best', borderpad=0.3,
                         shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                         markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)

        if show:
            plt.show()

    def make_prop_o_var_plot(self, s, num_obs, threshold=.95, show_it=True, last_plot=True):

        sum_s = sum(s.tolist())

        ss = s ** 2

        sum_ss = sum(ss.tolist())

        prop_list = list()

        found = False

        k = 0

        x1, y1, x2, y2, = 0, 0, 0, 0
        p_l, i_l = 0, 0
        found = False

        for i in range(1, num_obs + 1):
            perct = sum(ss[0:i]) / sum_ss
            # perct = sum(s[0:i]) / sum_s

            if np.around((perct * 100), 0) >= threshold*100 and not found:
                y2 = perct
                x2 = i
                x1 = i_l
                y1 = p_l
                found = True
            prop_list.append(perct)
            i_l = i
            p_l = perct

        if np.around(y2, 2) == .90:
            k_val = x2
        else:
            print('it is over 90%', x2)
            #vk_val = line_calc_x(x1, y1, x2, np.around(y2, 2), .9)

        single_vals = np.arange(1, num_obs + 1)

        if show_it:
            fig = plt.figure(figsize=(8, 5))
            plt.plot(single_vals, prop_list, 'ro-', linewidth=2)
            plt.title('Proportion of Variance, K should be {:d}'.format(x2))
            plt.xlabel('Eigenvectors')
            plt.ylabel('Prop. of var.')

            p90 = prop_list.index(y2)

            # plt.plot(k_val, prop_list[p90], 'bo')
            plt.plot(x2, prop_list[p90], 'bo')

            leg = plt.legend(['Eigenvectors vs. Prop. of Var.', '90% >=  variance'],
                             loc='best', borderpad=0.3,shadow=False, markerscale=0.4)
            leg.get_frame().set_alpha(0.4)
            #leg.draggable(state=True)

            if last_plot:
                plt.show()

        return x2


    def Groc(self, tpr, tnr):
        self.basic_plot(1-tnr, tpr)

    def gaussian_plot(self, xarrays, mus, stds, priors=[1, 1], verbose=False):
        for xarray, mu, std, prior in zip(xarrays, mus, stds, priors):
            Visualizer().basic_plot(xarray, generate_gaussian(xarray, mu, std, prior, verbose), xlabel='x',
                                    ylabel='prob',
                                    title='test gaussian', show=False, fig_num=1, m_label=[['a'], ['b']], legend=True)
        plt.show()
    # ================================================================================
    # ================================================================================
    # ====== TODO:                     std out methods                  ==============
    # ================================================================================
    # ================================================================================
    def string_padder(self,str='What Up Yo!', pstr=' ', addstr='Just Added', padl=20, right=True):
        if right:
            return str + '{:{}>{}s}'.format(addstr, pstr, padl)
        return str + '{:{}<{}s}'.format(addstr, pstr, padl)

    def border_maker(self, item, bsize=35):
        rs = ''
        for i in range(bsize):
            rs += item
        return rs

    def border_printer(self, border, padl=2):
        for i in range(padl):
            print(border)

    def create_label_string(self, label, border, lpad=4, lpstr=' ', b_size=35):
        # calculate border left over
        rpd = self.border_maker(lpstr, lpad)
        label = rpd + label + rpd
        b_left_over = b_size - len(label)
        if b_left_over%2 == 0:
            bleft = int(b_left_over/2)
            bright = int(b_left_over/2)
        else:
            bleft = int(np.around((b_left_over/2), 0))-1
            bright = int(np.around(b_left_over/2, 0))

        #return self.string_padder(str=border[0:bleft-(len(label))], pstr=lpstr, addstr=label, padl=lpad,
        return border[0:bleft] + label + border[0:bright]

    def block_label(self, label, lpad=4, lpstr=' ', border_marker=None, border_size=35, bpadl=2):
        if border_marker is not None:
            border =self.border_maker(border_marker, bsize=border_size)
            self.border_printer(border, padl=bpadl)
        else:
            border = self.border_maker('=', bsize=border_size)
            self.border_printer(border, padl=bpadl)

        print(self.create_label_string(label, border, lpad=lpad, lpstr=lpstr, b_size=border_size))

        if border_marker is not None:
            self.border_printer(self.border_maker(border_marker, bsize=border_size), padl=bpadl)
        else:
            self.border_printer(self.border_maker('=', bsize=border_size), padl=bpadl)

    def display_significance(self, feature_sig, features, verbose=False):
        """Takes """
        rd = {}
        for s, f in zip(feature_sig, features):
            rd[f] = s

        sorted_rd = dict(sorted(rd.items(), key=operator.itemgetter(1), reverse=True))
        if verbose:
            display_dic(sorted_rd)
        return sorted_rd

    def show_performance(self, scores, verbose=False, retpre=False):
        """displays a confusion matrix on std out"""
        true_sum = scores['tp'] + scores['tn']
        false_sum = scores['fp'] + scores['fn']
        sum = true_sum + false_sum

        # do this so we don't divde by zero
        tpfp = max(scores['tp']+scores['fp'], .00000001)
        tpfn = max(scores['tp']+scores['fn'], .00000001)
        precision = scores['tp']/tpfp
        recall = scores['tp']/tpfn
        accuracy = true_sum / sum
        #                 probability ot a true positive
        sensitivity = scores['tp'] / (scores['tp'] + scores['fn'])
        #                 probability ot a true negative
        specificity = scores['tn'] / (scores['tn'] + scores['fp'])
        if verbose:
            print('=====================================================')
            print('=====================================================')
            print('             |  predicted pos   |   predicted neg   |')
            print('----------------------------------------------------')
            print(' actual pos  |   {:d}            |   {: 3d}            |'.format(scores['tp'], scores['fn']))
            print('----------------------------------------------------')
            print(' actual neg  |   {:d}            |   {:d}            |'.format(scores['fp'], scores['tn']))
            print('-------------------------------------------------------------------')
            print('                                        Correct  |   {:d}'.format(true_sum))
            print('                                          Total  | % {:d}'.format(sum))
            print('                                                 | ------------------------')
            print('                                       Accuracy  | {:.2f}'.format(accuracy))
            print('                                      Precision  | {:.2f}'.format(precision))
            #print('                                         recall  | {:.2f}'.format(recall))
            print('                                    Sensitivity  | {:.2f}'.format(sensitivity))
            print('                                    Specificity  | {:.2f}'.format(specificity))
            print('=======================================================================================')
        if retpre:
            return accuracy, sum, sensitivity, specificity, precision

        return accuracy, sum, sensitivity, specificity

    def show_image(self, filename):
        """
            Can be used to display images to the screen
        :param filename:
        :return:
        """
        img = mpimg.imread(filename)
        plt.imshow(img)
        plt.show()

    def display_DT(self, estimator, features, classes, newimg='tree.png', tmpimg='tree.dot', precision=2):
        """plots a given decision tree"""
        from sklearn.tree import export_graphviz
        import io
        import pydotplus
        #graph = Source(export_graphviz(estimator, out_file=None
        #                                    , feature_names=features, class_names=['0', '1']
        #                                    , filled=True))
        #display(SVG(graph.pipe(format='svg')))
        # plot_tree(estimator, filled=True)
        # plt.show()
        # return

        # Export as dot file
        export_graphviz(estimator, out_file=tmpimg,
                                   feature_names=features,
                                   class_names=classes,
                                   rounded=True, proportion=False,
                                   precision=3, filled=True)
        #from subprocess import call
        #call(['dot', '-Tpng', tmpimg, '-o', newimg, '-Gdpi=600'])
        # os.system('dot -Tpng {} -o {}, -Gdpi=600'.format(tmpimg, newimg))
        # Display in python
        #import matplotlib.pyplot as plt

        # Draw graph
        #graph = graphviz.Source(dot_data)
        #dotfile = io.StringIO()
        graph = pydotplus.graph_from_dot_file(tmpimg)
        graph.write_png(newimg)
        print(graph)

        # Convert to png using system command (requires Graphviz)

        # plt.figure(figsize=(14, 18))
        # plt.imshow(plt.imread(newimg))
        # plt.axis('off')
        # plt.show()

        #from subprocess import call
        #os.system('dot -Tpng tmpimg -o newimg, -Gdpi=600')
        #self.show_image(newimg)

    def annotation_corre(self, dim, ax, matx, color='w', fontdict=None):
        if fontdict is None:
            fontdict = {
                'family': 'serif',
                'style': 'normal',
                'variant': 'normal',
                'weight': 'bold',
                #'size': 'large',
                'size': 15,
            }
        for i in range(len(dim)):
            for j in range(len(dim)):
                text = ax.text(j, i, np.around(matx[i, j], 3), ha='center', va='center', color=color, fontdict=fontdict)
        return

    def plot_img_matrix(self, data, xlabels='', ylabels='', title='Title_Plot_{}', fig_num=None, img_size=(24, 16),
                        save_image=None, sv_file=None, cmap='plasma', show_it=False, fsize=22, annote=True):
        fig, ax1,  = plt.subplots(1,1, figsize=img_size)
        #plt.figure(fig_num)
        #plt.xlabel(xlabels)

        plt.rcParams.update({'font.size': fsize})
        ax1.set_yticks(range(len(xlabels)))
        ax1.set_xticks(range(len(xlabels)))
        ax1.set_yticklabels(ylabels, fontdict={'size':fsize})
        ax1.set_xticklabels(xlabels, rotation='75', fontdict={'size':fsize})
        plt.title(title.format(1))
        ax1.imshow(data, cmap=cmap)
        #fig.tight_layout()
        self.annotation_corre(xlabels, ax1, data.values)
        if save_image is not None:
            plt.savefig(save_image)
        if show_it:
            plt.show()
        print(dir(fig.figure))

    def plot_dendrogram(self, corr, usecols, img_size=(24,24), title='title', show_it=False,
                        fsize=22):
        #corr = spearmanr(df0s).correlation
        from scipy.cluster import hierarchy
        fontdict = {
            'family': 'serif',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'bold',
            # 'size': 'large',
            'size': 15,
        }
        #plt.rcParams.update({'font.size': fsize})
        plt.rcParams.update({'font.size': fsize})
        corr = corr.values
        corr_linkage = hierarchy.ward(corr)
        fig, ax1, = plt.subplots(1, 1, figsize=img_size)
        dendro = hierarchy.dendrogram(corr_linkage, labels=usecols, ax=ax1,
                                      leaf_rotation=90)
        dendro_idx = np.arange(0, len(dendro['ivl']))
        ytck = list([""]*6)
        print('index', dendro_idx)
        print('dnd', dendro['ivl'])
        #ax1.set_xticks(dendro_idx)
        #ax1.set_yticks(dendro_idx)
        plt.title(title)
        ax1.set_xticklabels(dendro['ivl'], rotation='90', fontdict=fontdict)
        ax1.set_yticklabels(ytck)
        fig.tight_layout()
        #plt.savefig(plot_dir + r'\test.png')
        if show_it:
            plt.show()

    def plot_FI_PI(self, rf_clf, X_tr, y_tr, usecols, titles=('Plot 1', 'Plot 2'), top_num=-20,
                   verbose=False, reverse=True, n_repeats=20, random_state=42, fontdict=None, dpi=300,
                   figsize=(15, 15), title='RFI vs PI', files=None, src='', region='', save_it=False,
                   show_it=True, figname='fig_name.svg', title1='FI_ranking', figsize2=(10,6)):
        from _products._DEEPSOLAR_ import label_translation_dict
        from _products.ML_Tools import display_significance

        if fontdict is None:
            # title
            fontdict = {
                #'family': 'serif',
                'family': 'sans-serif',
                #'family': 'monospace',
                'style': 'normal',
                'variant': 'normal',
                'weight': 'heavy',
                'size': '12',
            }
        # labels
        fontdict2 = {
            'family': 'serif',
            #'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'medium',
            'size': '10',
        }

        # sub title
        fontdict3 = {
            # 'family': 'serif',
            'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'heavy',
            'size': '12',
        }

        # TODO: grab feature importance rankings
        feature_impz = rf_clf.feature_importances_
        feates2 = display_significance(feature_impz, usecols, verbose=verbose,reverse=reverse)
        pandas_excel_maker(files[0], params=feates2)
        # TODO: grab the permutation importances
        result = permutation_importance(rf_clf, X_tr, y_tr, n_repeats=n_repeats, random_state=random_state)
        perm_impz = display_significance(result.importances_mean, usecols, verbose=verbose, reverse=reverse)
        #pandas_excel_maker(files[1], params=perm_impz)
        # sort the indices
        perm_sorted_idx = result.importances_mean.argsort()

        tree_importance_sorted_idx = np.argsort(rf_clf.feature_importances_)
        #tree_indices = np.arange(0, len(rf_clf.feature_importances_)) + 0.001
        variables_in_set = np.array(usecols)
        lmt = max(-90, -len(feates2))
        plt.figure(figsize=figsize2, dpi=dpi)

        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,)
        yl = variables_in_set[tree_importance_sorted_idx]

        print('yl')
        print(yl)
        top_10v = list(feates2.values())[:-1 * lmt]
        top_10v.reverse()
        top_10l = list(feates2.keys())[:-1 * lmt]
        top_10l.reverse()
        print('labels ?')
        print(top_10l)
        # RF tree feature importance graph
        # ax1.barh(tree_indices[:-1*lmt],
        #         rf_clf.feature_importances_[tree_importance_sorted_idx][lmt:], height=0.7,)

        rfi_df = pd.read_csv(files[0])
        #top_10l = rfi_df['Variable'].tolist()[:-1*lmt]
        #top_10v = rfi_df['Avg_Importance'].tolist()[:-1*lmt]
        top_10l = rfi_df['Variable'].tolist()
        new_topl = [label_translation_dict[f] for f in top_10l]
        top_10l = new_topl
        top_10v = rfi_df['Avg_Importance'].tolist()
        # TODO: create the indices for the bar graph
        tree_indices = np.arange(0, len(top_10l)) + 0.4

        print('tree indices {}'.format(tree_indices))
        ax1.barh(tree_indices,
                 # ax1.barh(top_10l,
                 top_10v, height=0.4, )
        # ax1.set_yticklabels(data.feature_names[tree_importance_sorted_idx])
        # ax1.set_yticklabels(yl)
        # ax1.set_ylabel(list(feates2.keys())[:-1*lmt])
        ax1.set_yticks(tree_indices)
        ax1.set_yticklabels(top_10l, )
        ax1.set_ylim((0, len(tree_indices)))
        #ax1.set_title('Feature Importance: {}, region: {}'.format(src, region.upper()), fontdict=fontdict)
        ax1.set_title(title1, fontdict=fontdict)
        vset = [label_translation_dict[f]  for f in variables_in_set[perm_sorted_idx]]
        # Permutation importance box plot
        ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                    labels=vset, )
        #ax2.set_title('Top {} features for Permutation Importance '.format(-1 * lmt), x = -0.02)
        ax2.set_title(title, fontdict=fontdict, x=-.2)
        fig.tight_layout()
        if src == 'Xu':
            src = 'Dr. Xu\'s'
        if region == 'tva'.upper():
            x = .55
        if region == '7 State'.upper():
            x = .65
        if region == '13 State'.upper():
            x = .65
        fig.suptitle('{} Region, {} Model'.format(region, src), fontdict=fontdict, y=.02, x=.550,)
        #fig.suptitle('Variable Type: {}\n'.format(src)+title1 + ' ' + title, fontdict=fontdict, y = -1)
        #plt.title('RFI vs PI set {} accuracy {}'.format(src, np.around(accTS, 3)))
        #plt.title(title, fontdict=fontdict)

        if save_it:
            plt.savefig(figname, dpi=dpi)
        if show_it:
            plt.show()


    def plot_PI(self, rf_clf, X_tr, y_tr, usecols, titles=('Plot 1', 'Plot 2'), top=-20,
                   verbose=False, reverse=True, n_repeats=20, random_state=42, fontdict=None, dpi=300,
                   figsize=(15, 15), title='RFI vs PI', files=None, src='', region='', save_it=False,
                   show_it=True, figname='fig_name.svg', title1='FI_ranking', figsize2=(10,6), feates2='', label_translation_dict=None):
        #if label_translation_dict is None:
        #    from _products._DEEPSOLAR_ import label_translation_dict
        from _products.ML_Tools import display_significance
        import matplotlib
        font = {'size':12,
                'weight':'bold'}
        matplotlib.rc('font', **font)
        if fontdict is None:
            # title
            fontdict = {
                #'family': 'serif',
                'family': 'sans-serif',
                #'family': 'monospace',
                'style': 'normal',
                'variant': 'normal',
                'weight': 'heavy',
                'size': '15',
            }
        # labels
        fontdict2 = {
            'family': 'serif',
            #'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'medium',
            'size': '10',
        }

        # sub title
        fontdict3 = {
            # 'family': 'serif',
            'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'heavy',
            'size': '12',
        }

        # TODO: grab feature importance rankings
        #feature_impz = rf_clf.feature_importances_
        #feates2 = display_significance(feature_impz, usecols, verbose=verbose,reverse=reverse)
        #pandas_excel_maker(files[0], params=feates2)
        # TODO: grab the permutation importances
        result = permutation_importance(rf_clf, X_tr, y_tr, n_repeats=n_repeats, random_state=random_state)
        perm_impz = display_significance(result.importances_mean, usecols, verbose=verbose, reverse=reverse)
        #pandas_excel_maker(files[1], params=perm_impz)
        # sort the indices
        perm_sorted_idx = result.importances_mean.argsort()

        #tree_importance_sorted_idx = np.argsort(rf_clf.feature_importances_)
        #tree_indices = np.arange(0, len(rf_clf.feature_importances_)) + 0.001
        variables_in_set = np.array(usecols)
        lmt = max(top, -len(feates2))
        plt.figure(figsize=figsize2, dpi=dpi)

        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

        fig, ax2 = plt.subplots(1, 1, figsize=figsize,)
        #yl = variables_in_set[tree_importance_sorted_idx]
        yl = variables_in_set[perm_sorted_idx][top:]

        #print('yl')
        #print(yl)
        #top_10v = list(feates2.values())[:-1 * lmt]
        #top_10v.reverse()
        #top_10l = list(feates2.keys())[:-1 * lmt]
        #top_10l.reverse()
        #print('labels ?')
        #print(top_10l)
        # RF tree feature importance graph
        # ax1.barh(tree_indices[:-1*lmt],
        #         rf_clf.feature_importances_[tree_importance_sorted_idx][lmt:], height=0.7,)

        #rfi_df = pd.read_csv(files[0])
        #top_10l = rfi_df['Variable'].tolist()[:-1*lmt]
        #top_10v = rfi_df['Avg_Importance'].tolist()[:-1*lmt]
        #top_10l = rfi_df['Variable'].tolist()
        #new_topl = [label_translation_dict[f] for f in top_10l]
        #top_10l = new_topl
        #top_10v = rfi_df['Avg_Importance'].tolist()
        # TODO: create the indices for the bar graph
        #tree_indices = np.arange(0, len(top_10l)) + 0.4

        #print('tree indices {}'.format(tree_indices))
        #ax1.barh(tree_indices,
                 # ax1.barh(top_10l,
        #         top_10v, height=0.4, )
        # ax1.set_yticklabels(data.feature_names[tree_importance_sorted_idx])
        # ax1.set_yticklabels(yl)
        # ax1.set_ylabel(list(feates2.keys())[:-1*lmt])
        #ax1.set_yticks(tree_indices)
        #ax1.set_yticklabels(top_10l, )
        #ax1.set_ylim((0, len(tree_indices)))
        #ax1.set_title('Feature Importance: {}, region: {}'.format(src, region.upper()), fontdict=fontdict)
        #ax1.set_title(title1, fontdict=fontdict)
        #vset = [label_translation_dict[f]  for f in variables_in_set[perm_sorted_idx]]
        if label_translation_dict is None:
            vset = [f for f in yl]
        else:
            vset = [label_translation_dict[f]  for f in yl]
        #print('the supposed variables in the set')
        #print(vset)
        # Permutation importance box plot
        ax2.set_facecolor('xkcd:pale grey')
        if src == 'Xu':
            src = 'Reg-Selective'
        plt.xlabel('{} Model'.format(src), fontdict=fontdict)
        thing = result.importances[perm_sorted_idx][top:].T
        pi_avg = {}
        for var, vall in zip(vset, thing):
            pi_avg[var] = np.mean(vall)


        print('the length of the thing is ', len(thing))
        print('the len of the vars is ', len(vset))
        print('the averaged pi')
        for pi in pi_avg:
            print('{}: {}'.format(pi, pi_avg[pi]))
        ax2.boxplot(thing, vert=False,
                    labels=vset[top:], )
        #ax2.set_title('Top {} features for Permutation Importance '.format(-1 * lmt), x = -0.02)
        ax2.set_title(title.format(region, n_repeats), fontdict=fontdict, )
        fig.tight_layout()

        #print(' ****   ****   SEE ME\n{}'.format(result.importances[perm_sorted_idx].T))

        if region == 'tva'.upper():
            x = .55
        if region == '7 State'.upper():
            x = .65
        if region == '13 State'.upper():
            x = .65
        #fig.suptitle('{} Region, {} Model'.format(region, src), fontdict=fontdict, y=.02, x=.550,)
        #fig.suptitle('Variable Type: {}\n'.format(src)+title1 + ' ' + title, fontdict=fontdict, y = -1)
        #plt.title('RFI vs PI set {} accuracy {}'.format(src, np.around(accTS, 3)))
        #plt.title(title, fontdict=fontdict)

        if save_it:
            plt.savefig(figname, dpi=dpi)
        if show_it:
            print('should be seeing it')
            plt.show()
        return perm_impz


    def block_group_bar(self, filename, group_labels, reg, src, figsize=(10, 10), plot_name='{}_plot.svg',
                        accTS=None, null_acc=None, show_it=False, dpi=600, figsize2=(10,6), save_it=False,
                        ):

        # labels
        fontdict2 = {
            'family': 'serif',
            # 'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'medium',
            'size': '10',
        }
        if os.path.isfile(filename):
            df = pd.read_csv(filename, index_col='Block_Groups')
        else:
            dictt = {
                'Block_Groups': group_labels,
                'Acc': list([0] * len(group_labels)),
                'Imp': list([0] * len(group_labels)),
                'Nll': list([0] * len(group_labels)),
            }
            df = pd.DataFrame(dictt)
            df.to_csv(filename, index=False)
            df = pd.read_csv(filename, index_col='Block_Groups')
        # df['Block_Groups'] = bgs
        if src is None:
            src = 'ALL'
        if src == 'income employment':
            print('\n\n----------------------\n-----------------\n\n')
            src = 'inc/empl'
        if df.loc[src, 'Acc'] != 0:
            df.loc[src,'Acc'] = (df.loc[src,'Acc'] + accTS)/2
            df.loc[src,'Nll'] = (df.loc[src,'Nll'] + null_acc)/2
            #df.loc[src, 'Acc'] = ((accTS - null_acc) + df.loc[src, 'Acc']) / 2
            df.loc[src, 'Imp'] = (df.loc[src,'Imp'] + (accTS-null_acc))  / 2
        else:
            df.loc[src, 'Imp'] = (accTS - null_acc)
            df.loc[src, 'Acc'] = accTS
            df.loc[src, 'Nll'] = null_acc
        print('------------------------------------------------------------------------')
        print('{:s} avg. Accuracy: {:.2f}, Null: {:.2f}'.format(src, df.loc[src, 'Acc']*100, df.loc[src,'Nll']*100))
        print('------------------------------------------------------------------------')
        xv = list()
        for v in range(len(df)):
            if v == 0:
                xv.append(v)
            else:
                xv.append(v+.05)
        impv = df['Imp'].values.tolist()   # grab current values for improvement over null
        htsl = df['Acc'].values.tolist()   # grab current values for Accuracy
        impv = htsl
        # sub title
        fontdict3 = {
            # 'family': 'serif',
            'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'heavy',
            'size': '12',
        }
        plt.figure(figsize=figsize, dpi=dpi)
        fig3, ax3 = plt.subplots(1, 1, figsize=figsize2)        # create_subplot to get at fancy stuff
        #ax3.set_xticks(range(len(group_labels)))

        print('------------------')
        print('------------------')
        print('xv')
        print(xv)
        print('------------------')
        print('------------------')
        def get_accss(x):
            print('xacc',x)
            print('---------')
            l = list()
            #for xi in xv:
            #    l.append(htsl[xv.index(xi)])
            for xi in x:
                p = .05
                if xi[0] == 0:
                    p = 0
                l.append(htsl[xv.index(xi[0]+p)])
            return l[0], l[1]

        def get_impv(x):
            print('ximpv',x)
            print('---------')
            l = list()
            #for xi in xv:
            #    l.append(impv[xv.index(xi)])
            for xi in x:
                p = .05
                if xi[0] == 0:
                    p = 0
                l.append(htsl[impv.index(xi[0])])
            return l[0], l[1]
        #secax = ax3.secondary_yaxis('right', functions=(get_impv,
        #                                                get_accss))
        #secax.set_ylabel('Accuracy')
        ax3.set_xticks(xv)                                      # use list generated from for loop above to set xpostions of bars
        ax3.set_ylim(np.min(impv) - .5, np.max(impv) + .05)
        ax3.set_xticklabels(group_labels, rotation='horizontal', fontdict={'size': 10})   # let up the names for the bars
        ax3.bar(xv, impv, align='center', color='green',)                            # create the bar graph
        plt.title('Block group Predictive Accuracy: {}'.format(reg.upper()),
                  fontdict=fontdict3)
        ax3.set_facecolor('xkcd:light grey')
        # plt.xticks(list(range(len(bgs))), bgs,rotation='vertical')
        # ax3.xlabel('Block Group', fontdict={'size':15}, rotation='horizontal')
        #plt.ylabel('Predictive Accuracy', fontdict=fontdict2)
        plt.ylabel('Accuracy Improvment over Null', fontdict=fontdict2)

        df.to_csv(filename, index_label='Block_Groups', )

        for l in range(len(group_labels)):
            #if src == '':
            plt.text(xv[l]-.2, impv[l]+.0002, '{:.1f}'.format(df.loc[group_labels[l], 'Acc']*100), fontdict=fontdict2)
            #else:
            #    plt.text(xv[l] - .2, impv[l] + .0002, '{:.1f}'.format(df.loc[src, 'Acc'] * 100), fontdict=fontdict2)
        if save_it:
            fig3.savefig(plot_name)
        if show_it:
            plt.show()

    def plot_blocks_(self, filename, group_labels, reg, src, figsize=(10, 10), plot_name='{}_plot.svg',
                       show_it=False, dpi=600, figsize2=(10,6), save_it=False,
                        ):

        # labels
        fontdict2 = {
            'family': 'serif',
            # 'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'medium',
            'size': '10',
        }

        df = pd.read_csv(filename, index_col='Block_Groups')
        # df['Block_Groups'] = bgs
        if src is None:
            src = 'ALL'
        if src == 'income employment':
            src = 'inc/empl'


        xv = list()
        for v in range(len(group_labels)):
            if v == 0:
                xv.append(v)
            else:
                xv.append(v + .05)
        impv = df.loc[group_labels, 'Imp'].values.tolist()  # grab current values for improvement over null
        htsl = df.loc[group_labels, 'Acc'].values.tolist()  # grab current values for Accuracy
        # sub title
        fontdict3 = {
            # 'family': 'serif',
            'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'heavy',
            'size': '12',
        }
        plt.figure(figsize=figsize, dpi=dpi)
        fig3, ax3 = plt.subplots(1, 1, figsize=figsize2)  # create_subplot to get at fancy stuff
        # ax3.set_xticks(range(len(group_labels)))

        print('------------------')
        print('------------------')
        print('xv')
        print(xv)
        print('------------------')
        print('------------------')


        ax3.set_xticks(xv)  # use list generated from for loop above to set xpostions of bars
        ax3.set_xticklabels(group_labels, rotation='horizontal', fontdict={'size': 10})  # let up the names for the bars
        #fig.
        ax3.bar(xv, impv, align='center', color='green', )  # create the bar graph
        plt.title('Block group Predictive Accuracy: {}'.format(reg.upper()),
                  fontdict=fontdict3)
        ax3.set_facecolor('xkcd:light grey')
        # plt.xticks(list(range(len(bgs))), bgs,rotation='vertical')
        # ax3.xlabel('Block Group', fontdict={'size':15}, rotation='horizontal')
        # plt.ylabel('Predictive Accuracy', fontdict=fontdict2)
        plt.ylabel('Accuracy Improvment over Null', fontdict=fontdict2)

        df.to_csv(filename, index_label='Block_Groups', )

        for l in range(len(group_labels)):
            # if src == '':
            plt.text(xv[l] - .2, impv[l] + .0002, '{:.1f}'.format(df.loc[group_labels[l], 'Acc'] * 100),
                     fontdict=fontdict2)
            # else:
            #    plt.text(xv[l] - .2, impv[l] + .0002, '{:.1f}'.format(df.loc[src, 'Acc'] * 100), fontdict=fontdict2)
        if save_it:
            fig3.savefig(plot_name)
        if show_it:
            plt.show()


    def visualize_weights(self, model, layer_names, figsize, plot_names, feats,  dpi=600, cmap='plasma', annotate=None,
                          latent_size=None):

        colors = [
            [[0, 0, 0]],
            [[1, 0, 0]],
            [[0, 1, 0]],
            [[0, 0, 1]],
            [[.5, .005, .0]],
            [[.2, .02, .6]],
            [[.5, .82, .5]],
            [[.47, .09, .65]],
            [[.17, .02, .95]],
            [[.97, .03, .05]],
            [[.07, .29, .55]],
            [[.77, .89, .95]],
            [[.317, .41189, .35]],
        ]

        #imb2 = np.array([[[1, 1, 1]], [[1, .0, 0]], [[1, 1, 1]], [[0,1,0]], [[1,1,1]],[[0,0,1]], [[1,1,1]],])
        imb = list()
        added = 0
        vid_l = list()
        vid = int(np.floor(len(feats)/2))
        print('the vid {}'.format(vid))

        # for each node in encoding layer grab it?
        for f in range(len(feats)*latent_size):
            #if f ==  (len(feats) - latent_size) and f != 0 and added < latent_size:
            if f == vid and added < latent_size:
                vid_l.append(vid)
                vid += len(feats)
                imb.append(colors[added])
                added = (added + 1)%len(colors)
            else:
                imb.append(np.array([[1.0,1.0,1.0]]))

        imb = np.array(imb)

        #print('imb')
        #print(imb)
        #print('imb2')
        #print(imb2)
        #quit()
        layer_w_dict = {}

        fontdict = {
            # 'family': 'serif',
            'family': 'sans-serif',
            # 'family': 'monospace',
            'style': 'normal',
            'variant': 'normal',
            'weight': 'heavy',
            'size': '12',
        }

        print('there are {} total layers'.format(len(model.layers)))
        #quit()

        if False:
            # go through the layers of the model printing the weights and bias
            for lyr in model.layers:
                print('the layer weights')
                print(lyr.get_weights()[0])
                print(lyr.get_weights()[1])
                pass


        # set up a plot for the layers
        fig, axies = plt.subplots(1, len(model.layers)+1, figsize=figsize, )
        cnt = 1

        latent_axis = axies[1]
        latent_axis.imshow(imb)
        latent_axis.set_xticks([])
        latent_axis.set_yticks([])
        latent_axis.axis('off')

        plt.figure(figsize=figsize, dpi=dpi)
        # through each layer and axies 0, 2-N, all but the middle(encoding)
        for layer, ax in zip(model.layers[:len(model.layers)], list(axies[0:1]) + list(axies[2:])):
            # make a dictionary for this layers weights
            #imb = np.array([[[1, 1, 1]], [[1, .0, 0]], [[1, 1, 1]], [[0,1,0]], [[1,1,1]],[[0,0,1]],])
            layer_w_dict[cnt] = dict()
            #TODO: get the weights of the layer
            # if beyond input we dont know what the names are
            # so right now just make dummies
            if cnt > 1:
                feats = None

            # get the weights for the current layer
            weights = np.array(layer.get_weights()[0])

            labels = list()
            # figure out how many neurons for this layer
            neurons_in_lr = weights.shape[1]
            # figure out how many neurons we are being getting input from
            inputs_to_lr = weights.shape[0]
            if True:
                print('Each of the {} neurons takes inputs from {} inputs'.format(neurons_in_lr, inputs_to_lr))
            long_w = list()

            # TODO: set up the neuron label
            for c in range(neurons_in_lr):
                if feats is None:
                    labels += ['n{}_'.format(c) + str(f) for f in range(inputs_to_lr)]
                else:
                    labels += ['n{}_'.format(c) + '_' + str(f) for f in feats]

            c = 0
            # look at the weights for the neurons in the layer
            for neuron in range(weights.shape[1]):
                layer_w_dict[cnt][neuron] = dict()
                for inpt in weights[:, neuron]:
                    long_w.append(np.array(inpt))
                    layer_w_dict[cnt][neuron][labels[c]]=np.around(np.abs(inpt), 3)
                    c += 1
                layer_w_dict[cnt][neuron] = sort_dict(layer_w_dict[cnt][neuron],)
            long_w = np.array(long_w).reshape(len(long_w), 1)
            weights = long_w
            # print(weights)
            print(labels)
            img = ax.imshow(weights, cmap=cmap)
            if cnt == 1:
                vdx = 0
                for l in range(len(labels)):
                    if l != 0 and l%len(feats) == 0:
                        vdx += 1
                    ax.plot(list([.5, 15.5]), list([l, vid_l[vdx]]))
                    #print('plotted')
            #ax2.set_xticks(dendro_idx)
            ax.set_yticks(list(range(len(labels))))
            ax.set_yticklabels(list(labels))
            fig.colorbar(img, cmap=cmap)
            fig.tight_layout(pad=.0)
            fig.patch.set_visible(False)
            #ax.axis('off')
            ax.patch.set_visible(False)
            ax.set_xticks([])
            plt.ylabel('')
            cnt += 1
            #ax.set_title('Layer {}: {} weights'.format(cnt, layer_names[cnt-1]))
            if annotate is not None:
                annotate((len(weights), 1), ax, weights,)
        return layer_w_dict
        #plt.show()



    def display_weights_keras(self, rd):
        for l in rd:
            print('====================================================================================')
            print('====================================================================================')
            print('Layer: {}'.format(l))
            for n in rd[l]:
                print('\t\tNeuron: {}'.format(n))
                for f in rd[l][n]:
                    print('\t\t\t\t{:>s}-:{:.3f}'.format(f, rd[l][n][f]))
            print('====================================================================================')
            print('====================================================================================')

        return