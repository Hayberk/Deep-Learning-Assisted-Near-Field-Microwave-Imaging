import os
import sys
import numpy as np
import time
import skrf as rf
from skimage import draw
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
#alttaki path'i kendi HFSS'inizin kurulu oldugu diske gore guncelleyin
sys.path.append('C:/Program Files/AnsysEM/v222/Win64/PythonFiles/DesktopPlugin')
import ScriptEnv


class ImagingAlgorithm(object):

    def __init__(self, pixel=64, antenna=16, freq=3e9, sweep=5, start_pos_x=-0.64, start_pos_y=-0.64, obj1=None, obj2=None, datafile1=None, datafile2=None):
        self.pixel = pixel
        self.antenna = antenna
        self.freq = freq
        self.sweep = sweep
        self.start_pos_x = start_pos_x
        self.start_pos_y = start_pos_y
        self.scaling = 2*abs(start_pos_x)/pixel
        self.wavelength = 3e8 / freq
        self.k = (2 * np.pi * freq) / 3e8
        self.max_perm = 20
        self.deg = (360 / antenna) * (np.pi / 180)
        self.obj1 = obj1
        self.obj2 = obj2
        self.datafile1 = datafile1
        self.datafile2 = datafile2
        self.ground_truth = None
        self.image = None
        self.path_img = os.path.join(os.getcwd(), "images")
        os.makedirs(self.path_img, exist_ok=True)
        self.path_img_pil = os.path.join(os.getcwd(), "images_PIL")
        os.makedirs(self.path_img_pil, exist_ok=True)
        self.path_gt_pil = os.path.join(os.getcwd(), "ground_truth_PIL")
        os.makedirs(self.path_gt_pil, exist_ok=True)
        self.val = os.path.join(os.getcwd(), "values")
        os.makedirs(self.val, exist_ok=True)

    def create_image(self, a):

        def compute_image():
            theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            a_x = 0.15 * np.cos(np.pi / 2 - theta)
            a_y = 0.15 * np.sin(np.pi / 2 - theta)

            x = np.linspace(self.start_pos_x, abs(self.start_pos_x), self.pixel)
            y = np.linspace(self.start_pos_y, abs(self.start_pos_y), self.pixel)
            p_x, p_y = np.meshgrid(x, y)

            arr = np.zeros((self.pixel, self.pixel), dtype=complex)

            s_matrix1 = rf.Network(self.datafile1)
            s_matrix2 = rf.Network(self.datafile2)

            for x in range(self.sweep):
                s_matrix_diff = s_matrix1.s[x, :, :] - s_matrix2.s[x, :, :]

                for n in range(self.antenna):
                    for m in range(self.antenna):
                        dist_q = np.sqrt((p_x - a_x[n]) ** 2 + (p_y - a_y[n]) ** 2)
                        dist_p = np.sqrt((p_x - a_x[m]) ** 2 + (p_y - a_y[m]) ** 2)
                        exp_q = np.exp(-1j * self.k * dist_q) / (4 * np.pi * dist_q)
                        exp_p = np.exp(1j * self.k * dist_p) / (4 * np.pi * dist_p)
                        arr += s_matrix_diff[n, m] * exp_q * exp_p

            self.image = abs(arr / self.sweep)
            self.image = np.flipud(self.image)

        def save_image(a):
            fig, ax = plt.subplots()
            extent = [self.start_pos_x, self.start_pos_x + self.pixel * self.scaling, self.start_pos_y,  self.start_pos_y + self.pixel * self.scaling]
            im = ax.imshow(self.image, cmap='jet', origin='upper', extent=extent)
            ax.set_xlim(self.start_pos_x, self.start_pos_x + self.pixel * self.scaling)
            ax.set_ylim(self.start_pos_y - self.pixel * self.scaling, self.start_pos_y)
            fig.colorbar(im, ax=ax, ticks=np.linspace(self.image.min(), self.image.max(), 10))
            secax_x = ax.secondary_xaxis('bottom')
            secax_y = ax.secondary_yaxis('left')
            secax_x.set_xlabel('X [m]')
            secax_y.set_ylabel('Y [m]')
            plt.title(f"x1={self.obj1[0]},y1={self.obj1[1]},r1={self.obj1[2]},e1={self.obj1[3]},g1={self.obj1[4]}\nx2={self.obj2[0]},y2={self.obj2[1]},r2={self.obj2[2]},e2={self.obj2[3]},g2={self.obj2[4]}")
            plt.savefig(os.path.join(self.path_img, "x{:04d}.png".format(a)))
            plt.show()
            plt.close()

        def save_image_pil(a):
            image_pil = np.round((255 / 0.1) * self.image).astype(np.uint8)
            plt.imshow(image_pil, cmap='gray', origin='upper', vmin=0, vmax=255)
            plt.axis('off')
            plt.show()
            plt.close()
            im1 = Image.fromarray(image_pil)
            im1.save(os.path.join(self.path_img_pil, "x{:04d}.png".format(a)))

        def create_ground_truth(image):
            ground_truth = np.zeros((self.pixel, self.pixel), dtype=np.uint8)
            rr1, cc1, rr2, cc2 = 0, 0, 0, 0
            rr1, cc1 = draw.disk((np.round((abs(self.start_pos_x) - self.obj1[1] / 1000) / self.scaling), np.round((abs(self.start_pos_x - self.obj1[0] / 1000)) / self.scaling)),np.round((self.obj1[2] / 1000) / self.scaling), shape=ground_truth.shape)
            rr2, cc2 = draw.disk((np.round((abs(self.start_pos_x) - self.obj2[1] / 1000) / self.scaling), np.round((abs(self.start_pos_x - self.obj2[0] / 1000)) / self.scaling)),np.round((self.obj2[2] / 1000) / self.scaling), shape=ground_truth.shape)
            print(f"mean1:{image[rr1, cc1].mean()}")
            ground_truth[rr1, cc1] = np.round(255 / (self.max_perm-1) * (self.obj1[3] - 1), 0)
            ground_truth[rr2, cc2] = np.round(255 / (self.max_perm-1) * (self.obj2[3] - 1), 0)
            self.ground_truth = ground_truth

        def save_ground_truth(a):
            plt.figure()
            plt.imshow(self.ground_truth, cmap='gray', origin='upper', vmin=0, vmax=255)
            plt.axis('off')
            plt.show()
            plt.close()
            im1 = Image.fromarray(self.ground_truth)
            im1.save(os.path.join(self.path_gt_pil, "x{:04d}.png".format(a)))

        compute_image()
        save_image(a)
        save_image_pil(a)
        create_ground_truth(self.image)
        save_ground_truth(a)


class DatasetCreator(object):

    def __init__(self, amount):
        self.amount = amount
        self.obj1 = [0, 0, 0, 0]
        self.obj2 = [0, 0, 0, 0]
        self.max_perm = 20
        self.path_tf = os.path.join(os.getcwd(), "touchstone_files")
        os.makedirs(self.path_tf, exist_ok=True)

    def generate(self):

        def create_objects(num):
            for z in range(1, num + 1):
                perm = np.random.randint(2, self.max_perm+1)
                x1 = np.random.randint(-40, 41)
                y1 = np.random.randint(-40, 41)
                r1 = np.random.randint(10, 21)
                if z == 1:
                    self.obj1[0] = x1
                    self.obj1[1] = y1
                    self.obj1[2] = r1
                    self.obj1[3] = perm
                if z == 2:
                    d = np.sqrt(np.power((x1 - self.obj1[0]), 2) + np.power((y1 - self.obj1[1]), 2))
                    while d < (r1 + self.obj1[2] + 50):
                        x1 = np.random.randint(-40, 41)
                        y1 = np.random.randint(-40, 41)
                        r1 = np.random.randint(10, 21)
                        d = np.sqrt(np.power((x1 - self.obj1[0]), 2) + np.power((y1 - self.obj1[1]), 2))
                    self.obj2[0] = x1
                    self.obj2[1] = y1
                    self.obj2[2] = r1
                    self.obj2[3] = perm
                oDefinitionManager = oProject.GetDefinitionManager()
                oDefinitionManager.EditMaterial(f"test{z}",
                                                [
                                                    f"NAME:test{z}",
                                                    "CoordinateSystemType:=", "Cartesian",
                                                    "BulkOrSurfaceType:=", 1,
                                                    [
                                                        "NAME:PhysicsTypes",
                                                        "set:="		, ["Electromagnetic"]
                                                    ],
                                                    "permittivity:="	, f"{perm}"
                                                ])
                oEditor = oDesign.SetActiveEditor("3D Modeler")
                oEditor.CreateSphere(
                    [
                        "NAME:SphereParameters",
                        "XCenter:="		, f"{x1}mm",
                        "YCenter:="		, f"{y1}mm",
                        "ZCenter:="		, "0mm",
                        "Radius:="		, f"{r1}mm"
                    ],
                    [
                        "NAME:Attributes",
                        "Name:="		, f"Sphere{z}",
                        "Flags:="		, "",
                        "Color:="		, "(143 175 143)",
                        "Transparency:="	, 0,
                        "PartCoordinateSystem:=", "Global",
                        "UDMId:="		, "",
                        "MaterialValue:="	, "\"vacuum\"",
                        "SurfaceMaterialValue:=", "\"\"",
                        "SolveInside:="		, True,
                        "ShellElement:="	, False,
                        "ShellElementThickness:=", "0mm",
                        "IsMaterialEditable:="	, True,
                        "UseMaterialAppearance:=", False,
                        "IsLightweight:="	, False
                    ])
                oEditor.AssignMaterial(
                    [
                        "NAME:Selections",
                        "AllowRegionDependentPartSelectionForPMLCreation:=", True,
                        "AllowRegionSelectionForPMLCreation:=", True,
                        "Selections:="		, f"Sphere{z}"
                    ],
                    [
                        "NAME:Attributes",
                        "MaterialValue:="	, f"\"test{z}\"",
                        "SolveInside:="		, True,
                        "ShellElement:="	, False,
                        "ShellElementThickness:=", "nan ",
                        "IsMaterialEditable:="	, True,
                        "UseMaterialAppearance:=", False,
                        "IsLightweight:="	, False
                    ])
            print(f"x1={self.obj1[0]},y1={self.obj1[1]},r1={self.obj1[2]},perm1={self.obj1[3]},geo={self.obj1[4]}")
            print(f"x2={self.obj2[0]},y1={self.obj2[1]},r1={self.obj2[2]},perm1={self.obj2[3]},geo={self.obj2[4]}")

        def analyze(a):
            oDesign.Analyze("SweepAvg")
            oProject.Save()
            oModule = oDesign.GetModule("Solutions")
            oModule.ExportNetworkData(
                "deg=\'0.39269908169872414rad\' dL=\'44.45mm\' dR=\'1.5mm\' gL=\'1mm\' wl=\'150mm\'",
                ["SweepAvg:SweepAvg"], 3, "{}/x{:04d}.s16p".format(self.path_tf, a),
                [
                    "All"
                ], True, 50, "S", -1, 0, 15, True, True, False)

        def delete_objects(num):
            for i in range(1, num+1):
                oEditor.Delete(
                    [
                        "NAME:Selections",
                        "Selections:="		, f"Sphere{i}",
                    ])

        for a in range(self.amount):
            num = 2
            delete_objects(num)
            create_objects(num)
            analyze(a)
            img = ImagingAlgorithm(obj1=self.obj1, obj2=self.obj2, datafile1="{}/x{:04d}.s16p".format(self.path_tf, a), datafile2=os.path.join(os.getcwd(), "Sweep_empty_150mm.s16p"))
            img.create_image(a)
            delete_objects(num)


if __name__ == '__main__':
    start = time.time()
    project_file = os.path.join(os.getcwd(), 'antenna_array_testing.aedt')
    ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
    oDesktop.RestoreWindow()
    oDesktop.OpenProject(project_file)
    oProject = oDesktop.SetActiveProject("antenna_array_testing")
    oDesign = oProject.SetActiveDesign("HFSSDesign1")
    oEditor = oDesign.SetActiveEditor("3D Modeler")
    HFSS = DatasetCreator(amount=10)
    HFSS.generate()
    end = time.time() - start
    print(f"\nProcessing time: {end} seconds")
