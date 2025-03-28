from random import uniform, normalvariate
from math import log

# univariate continuous uniform proposal: sampler & log-pdf
def unif_proposal_log_pdf(from_smpl, to_smpl, lower_lim, upper_lim, window):
    "return: scala"
    from_smpl = from_smpl[0]
    to_smpl = to_smpl[0]
    applied_window = [max(lower_lim, from_smpl-window/2), min(upper_lim, from_smpl+window/2)]
    if to_smpl<applied_window[0] or to_smpl>applied_window[1]:
        # return -inf
        raise ValueError("to_smpl has an unacceptable value")
    else:
        applied_window_len = applied_window[1] - applied_window[0]
        # return 1/applied_window_len
        return -log(applied_window_len)

def unif_proposal_sampler(from_smpl, lower_lim, upper_lim, window):
    "return: [u]"
    from_smpl = from_smpl[0]
    applied_window = [max(lower_lim, from_smpl-window/2), min(upper_lim, from_smpl+window/2)]
    new = [uniform(applied_window[0], applied_window[1])]
    return new

def multivar_unif_proposal_log_pdf(from_smpl, to_smpl, lower_lim, upper_lim, window):
    "return: scala"
    val = 0
    for fs, ts, ll, ul, wnd in zip(from_smpl, to_smpl, lower_lim, upper_lim, window):
        new_i = unif_proposal_log_pdf([fs], [ts], ll, ul, wnd)
        val += new_i
    return val


def multivar_unif_proposal_sampler(from_smpl, lower_lim, upper_lim, window):
    "arg window/lower_lim/upper_lim: the same dim with from_smpl, return: [u]"
    new = []
    for fs, ll, ul, wnd in zip(from_smpl, lower_lim, upper_lim, window):
        new_i = unif_proposal_sampler([fs], ll, ul, wnd)
        new.append(new_i[0])
    return new

# univariate continuous uniform, spherical proposal: sampler & log-pdf
def unif_spherical_proposal_sampler(from_smpl, lower_lim, upper_lim, window):
    "return: [u]. Symmetric proposal scheme."
    from_smpl = from_smpl[0]
    applied_window = (from_smpl-window/2, from_smpl+window/2)
    support_length = upper_lim - lower_lim
    new_val = uniform(applied_window[0], applied_window[1])
    if new_val < lower_lim:
        new_val = new_val + support_length
    elif new_val > upper_lim:
        new_val = new_val - support_length
    return [new_val]

def unif_spherical_proposal_log_pdf(from_smpl, to_smpl):
    #symmetric
    return 0


# multivariate normal proposal: sampler & log-pdf
def gaussian_proposal_log_pdf(from_smpl, to_smpl):
    # symmetric
    return 0

def gaussian_proposal_sampler(from_smpl, proposal_sd: list):
    if len(from_smpl) != len(proposal_sd):
        raise ValueError("the dimensions of a MCMC sample and proposal sd list should be matched." )
    new = [normalvariate(x, s) for x,s in zip(from_smpl, proposal_sd)]
    return new