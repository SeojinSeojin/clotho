"""Measured validity gates, not simulated detector performance."""
import json
import common as C

def main():
    plt=C.style();root=C.rp('llama')/'strong_downstream_preflight';r={s:json.loads((root/f'{s}.json').read_text()) for s in ['HDFS','BGL']}
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    ax=axes[0];ax.bar(['Incomplete','Complete normal','Complete anomalous'],[4049,0,39],color=['#999999','#2a78d6','#eb6834']);ax.set_yscale('symlog',linthresh=1);ax.set_ylabel('HDFS blocks (symlog scale)');ax.set_title('Saved parser coverage: complete blocks')
    for i,v in enumerate([4049,0,39]):ax.text(i,max(v,0)+.5,str(v),ha='center',va='bottom')
    ax.set_xticks([0,1,2]);ax.set_xticklabels(['Incomplete','Complete\nnormal','Complete\nanomalous'],fontsize=8)
    ax=axes[1]
    for j,k in enumerate(['train','validation','test']):
        z=r['BGL']['splits'][k];ax.bar(j,z['normal'],label='Normal' if j==0 else None,color='#2a78d6');ax.bar(j,z['anomalous'],bottom=z['normal'],label='Anomalous' if j==0 else None,color='#eb6834');ax.text(j,z['normal']+z['anomalous']+25,f"{z['anomalous']} anomalous",ha='center',fontsize=8)
    ax.set_xticks(range(3));ax.set_xticklabels(['Train','Validation','Test']);ax.set_ylabel('BGL nonoverlapping 20-line windows');ax.set_title('Frozen 60/20/20 chronology');ax.legend(fontsize=8);ax.set_ylim(0,1650)
    fig.tight_layout();dst=C.PAPER/'figures';fig.savefig(dst/'strong_downstream_preflight.pdf',bbox_inches='tight');(dst/'strong_downstream_preflight.tex').write_text('Full-file coverage audit of saved 50,000-line parser assignments. HDFS has only39 complete blocks, all anomalous; BGL frozen test haszero positive windows. These are deterministic unit counts, not detector scores; no review-recovery curve is estimable.\n');plt.close(fig)
if __name__=='__main__':main()
