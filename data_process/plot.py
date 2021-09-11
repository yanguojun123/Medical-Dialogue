import matplotlib.pylab as plt
from pylab import *
def wk_plot():
    y1 = [10.17, 12.51, 13.14, 13.13, 13.20, 13.16]
    y2 = [15.86, 19.58, 20.51, 23.41, 23.79, 23.02]
    y3 = [8.81, 8.97, 8.31, 9.89, 10.03, 9.48]
    x = [0, 1, 3, 5, 7, 9]

    fig, ax1 = plt.subplots()  # Create a window using subplots()

    # mp.legend(loc=7)

    ax2 = ax1.twinx()  # Create a second axis
    lns1 = ax1.plot(x, y2, 'o-', c='lightskyblue', label='NLU', linewidth=1)
    lns3 = ax2.plot(x, y1, 'o-', c='goldenrod', label='NLG', linewidth=1)

    # mp.legend(loc=7)
    lns2 = ax1.plot(x, y3, 'o-', c='seagreen', label='DPL', linewidth=1)
    # mp.legend(loc=7,labelspacing=1)
    ax1.set_ylim((8, 25))
    ax2.set_ylim((10, 14))
    # ax1.set_xticks(x)  # Set scale
    ax1.set_xticklabels(['0', '1', '3', '5', '7', '9'])
    ax2.set_xticklabels(['0', '1', '3', '5', '7', '9'])
    # for tick in ax1.get_xticklabels():
    #     print(tick)
    #     tick.set_rotation(180)
    # for tick in ax2.get_xticklabels():
    #     print(tick)
    #     tick.set_rotation(180)
    # ax1.set_xlim(0,10)
    # ax1.set_xticks(rotation=90)
    # ax1.set_xlabel('#dialogues(x10^5)', horizontalalignment='right',fontdict={'family': 'Times New Roman', 'size': 13})
    ax1.set_xlabel(r'#dialogues($\times 10^5$)', fontdict={'family': 'Times New Roman', 'size': 13})
    ax2.set_ylabel(r'BLEU1(\%)', fontdict={'family': 'Times New Roman', 'size': 10})
    ax1.set_ylabel(r'Combination(\%)', fontdict={'family': 'Times New Roman', 'size': 13})
    # ax2.set_ylabel('y2', size=18)
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0, fontsize='small')
    # ax1.legend(lns, labs, bbox_to_anchor=(0.91,0.08),loc=4, ncol=3,fontsize='small',bbox_transform=fig.transFigure)
    # plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.gcf().autofmt_xdate()  # Automatically adapt to scale line density, including x-axis and y-axis
    # plt.xticks(rotation=0)
    plt.show()

def NP():
    fig=plt.figure()

    labels = ['all','alias','trans','random','none']

    plt.subplot(131)
    number = [37.03, 37.02, 36.81, 36.31, 36.41]
    index = ['all', 'alias', 'trans', 'random', 'none']
    plt.bar(index, number,color=['lightsalmon','wheat','yellowgreen','seagreen','skyblue'])
    plt.ylabel('Combination(%)',fontdict={'family': 'Times New Roman', 'size': 10})
    #plt.xlabel('methods')
    plt.ylim((36, 37.1))
    plt.yticks(fontproperties='Times New Roman', size=10)
    plt.xticks([])
    plt.title('NLU',fontdict={'family': 'Times New Roman', 'size': 11})
    plt.legend(labels,bbox_to_anchor=(0.91,0.08),loc=4, ncol=3,fontsize='small',bbox_transform=fig.transFigure)
    '''for ax in plt.axes:
        ax.set_xticks([])'''

    plt.subplot(132)
    number = [23.82, 23.10, 24.22, 24.83, 23.44]
    index = ['all', 'alias', 'trans', 'random', 'none']
    plt.bar(index, number, color=['lightsalmon','wheat','yellowgreen','seagreen','skyblue'])
    #plt.ylabel('Combination',fontdict={'family': 'Times New Roman', 'size': 13})
    #plt.xlabel('methods')
    plt.ylim((23, 25))
    plt.ylabel('Combination(%)', fontdict={'family': 'Times New Roman', 'size': 10})
    plt.yticks(fontproperties='Times New Roman', size=10)
    plt.xticks([])
    plt.title('DPL', fontdict={'family': 'Times New Roman', 'size': 11})


    plt.subplot(133)
    number = [26.54,26.55,27.38,25.05,25.97]
    index = ['all', 'alias', 'trans', 'random', 'none']
    #index = ['1', '2', '3', '4', '5']
    plt.bar(index, number, color=['lightsalmon','wheat','yellowgreen','seagreen','skyblue'])
    plt.ylabel('BLEU1(%)',fontdict={'family': 'Times New Roman', 'size': 10})
    #plt.xlabel('methods')
    plt.ylim((25, 27.5))
    plt.xticks([])
    plt.yticks(fontproperties='Times New Roman', size=10)
    plt.title('NLG',fontdict={'family': 'Times New Roman', 'size': 11})

    #plt.yticks([35+x*0.2 for x in range(15)])'
    #plt.legend(loc=0)
    #fig.legend(labels, loc='upper left', bbox_to_anchor=(0,1.02), ncol=5, bbox_transform=fig.transFigure)
    plt.tight_layout()  # Automatically adjust subgraph spacing

    plt.show()

def intent_slot():
    fig = plt.figure()

    plt.subplot(121)
    x = ['Informing', 'Inquiring', 'Recommendation', 'Chitchat', 'Others', 'QA', 'Diagnosis']
    y = [15904, 10395, 5209, 4806, 2779, 2481, 507]

    rects = plt.barh(range(len(y)), y, height=0.3, color='orange')
    plt.yticks(range(len(y)), x, rotation=0)
    # pyplot.grid(alpha=0.3)
    for x, y1 in zip(y, range(len(y))):
        plt.text(x + 2000, y1 - 0.1, '%.0f' % x, ha='center', va='bottom', fontsize=8)
    plt.title("intent distribution in utterance-level")
    # pyplot.show()
    '''for ax in plt.axes:
        ax.set_xticks([])'''

    plt.subplot(122)
    x = ["symptom", "treatment", "disease", "medicine", "time", "precaution", "other", "pathogeny", "check_item"
        , "effect", "disease-history", "degree", "side-effect", "range-body", "frequency", "medicine-category"
        , "department", "dose", "medical-place", "temperature"]
    y = [559, 407, 276, 268, 205, 175, 139, 108, 77, 50, 38, 36, 26, 24, 22, 20, 16, 12, 8, 2]

    rects = plt.barh(range(len(y)), y, height=0.3, color='orange')
    plt.yticks(range(len(y)), x, rotation=0)
    for x, y1 in zip(y, range(len(y))):
        plt.text(x + 50, y1 - 0.2, '%.0f' % x, ha='center', va='bottom',
                    fontsize=8)
    plt.title("slot entity")
    # pyplot.show()
    '''for ax in plt.axes:
        ax.set_xticks([])'''

    plt.tight_layout()  # Automatically adjust subgraph spacing

    plt.show()

if __name__ == '__main__':
    wk_plot()
    #NP()
    #intent_slot()