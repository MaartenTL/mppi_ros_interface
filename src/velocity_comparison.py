#!/usr/bin/env python3
import rospy, csv, os
from std_msgs.msg import Float32
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

class Recorder:
    def __init__(self, car, outdir):
        self.car = car
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.rows = []  # list of dicts per sample

        # last-sample cache so we can align by arrival time
        self.last = {
            't': None,
            'vx_gt': None, 'vy_gt': None, 'w_gt': None,
            'vx_fd': None, 'vy_fd': None, 'w_fd': None,
            'vx_ls': None, 'vy_ls': None, 'w_ls': None,
            'vx_ls_old': None, 'vy_ls_old': None, 'w_ls_old': None,
        }

        # subscribe GT
        rospy.Subscriber(f"/vx_{car}", Float32, self._mkcb('vx_gt'))
        rospy.Subscriber(f"/vy_{car}", Float32, self._mkcb('vy_gt'))
        rospy.Subscriber(f"/omega_{car}", Float32, self._mkcb('w_gt'))
        # FD
        rospy.Subscriber("vx_est_fd", Float32, self._mkcb('vx_fd'))
        rospy.Subscriber("vy_est_fd", Float32, self._mkcb('vy_fd'))
        rospy.Subscriber("omega_est_fd", Float32, self._mkcb('w_fd'))
        # LS
        rospy.Subscriber("vx_est_ls", Float32, self._mkcb('vx_ls'))
        rospy.Subscriber("vy_est_ls", Float32, self._mkcb('vy_ls'))
        rospy.Subscriber("omega_est_ls", Float32, self._mkcb('w_ls'))
        # AB
        rospy.Subscriber(f"vx_est_{car}", Float32, self._mkcb('vx_ls_old'))
        rospy.Subscriber(f"vy_est_{car}", Float32, self._mkcb('vy_ls_old'))
        rospy.Subscriber(f"omega_est_{car}", Float32, self._mkcb('w_ls_old'))

    def _mkcb(self, key):
        def _cb(msg):
            t = rospy.get_time()
            self.last['t'] = t
            self.last[key] = float(msg.data)
            # Snapshot into rows at each message arrival (simple union sampling)
            self.rows.append({
                't': t,
                'vx_gt': self.last['vx_gt'], 'vy_gt': self.last['vy_gt'], 'w_gt': self.last['w_gt'],
                'vx_fd': self.last['vx_fd'], 'vy_fd': self.last['vy_fd'], 'w_fd': self.last['w_fd'],
                'vx_ls': self.last['vx_ls'], 'vy_ls': self.last['vy_ls'], 'w_ls': self.last['w_ls'],
                'vx_ls_old': self.last['vx_ls_old'], 'vy_ls_old': self.last['vy_ls_old'], 'w_ls_old': self.last['w_ls_old'],
            })
        return _cb

    def save(self):
        if not self.rows:
            rospy.logwarn("No data recorded.")
            return
        # normalise time to t0 = first sample
        t0 = self.rows[0]['t']
        for r in self.rows:
            r['t'] = r['t'] - t0

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.outdir, f"vel_compare_car{self.car}_{stamp}.csv")
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            w.writeheader(); w.writerows(self.rows)
        rospy.loginfo(f"Wrote CSV: {csv_path}")

        # load to numpy (some entries None → mask)
        def col(name):
            return np.array([r[name] if r[name] is not None else np.nan for r in self.rows], float)
        T = col('t')

        def plot_one(sig, ylabel):
            gt = col(f"{sig}_gt")
            fd = col(f"{sig}_fd")
            ls = col(f"{sig}_ls")
            ls_old = col(f"{sig}_ls_old")

            plt.figure(figsize=(10,4))
            plt.plot(T, gt, label=f"{sig} GT")
            plt.plot(T, fd, label=f"{sig} FD", alpha=0.9)
            plt.plot(T, ls, label=f"{sig} LS", alpha=0.9)
            plt.plot(T, ls_old, label=f"{sig} LS OLD", alpha =0.9)
            plt.xlabel("time [s]"); plt.ylabel(ylabel); plt.title(f"{sig} comparison")
            plt.legend(); plt.tight_layout()
            out = os.path.join(self.outdir, f"{sig}_compare_car{self.car}_{stamp}.png")
            plt.savefig(out, dpi=150)
            rospy.loginfo(f"Wrote plot: {out}")
            plt.show()
            # RMSEs (ignore NaNs)
            def rmse(a,b):
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 5: return np.nan
                return np.sqrt(np.mean((a[m]-b[m])**2))
            rospy.loginfo("RMSE vs GT | %s: FD=%.4f  LS=%.4f LS_OLD=%.4f",
                          sig, rmse(gt,fd), rmse(gt,ls), rmse(gt,ls_old))

        plot_one('vx', 'm/s')
        plot_one('vy', 'm/s')
        plot_one('w',  'rad/s')

if __name__ == "__main__":
    rospy.init_node("record_vel_compare", anonymous=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="/home/maarten/Documents/Thesis/log_Dart/vel_compare")
    args, _ = ap.parse_known_args()

    rec = Recorder(args.car, args.outdir)
    rospy.loginfo("Recording… Ctrl+C to stop and write outputs.")
    try:
        rospy.spin()
    finally:
        rec.save()
        plt.show()
