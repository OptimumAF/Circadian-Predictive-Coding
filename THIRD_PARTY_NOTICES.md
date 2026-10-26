# Component and data notices

Scope: the CPU memory demo distribution, optional native PickCube/PushCube path
and separate Choice smoke, reconciled 2026-10-22. This notice separates this project's material from
separately installed simulator code and assets. It does not replace their licenses.

## Project code and generated numeric data

Project code is provided under the accompanying [MIT license](LICENSE).
The bundled `examples/taskspace_memory/episodes.json` (installed as
`src.infra.taskspace_demo_data/episodes.json`) contains locally generated numeric
TRAIN observations/actions and recorded trajectories. Its existing project MIT
grant and provenance remain in the accompanying resource `README.md`.
No simulator meshes, textures, images, SDK binaries or pretrained weights are
included in that bundle. This statement does not assign MIT to upstream assets
or determine permissions for every downstream use of simulator-derived data.

The additional `continuous_comparison_data/comparison-summary.json` and
`continuous_comparison_data/controls.json` resources are
project-generated numeric comparison evidence under the project MIT notice.
Its original results, adverse directions and provenance are retained; it is a
saved report rather than training inputs or a third-party dataset.

The `ticket_workflow_data/{bp,pc,neutral_cpc,gated_cpc}.json` resources are exact
project-generated synthetic ticket task reports under the accompanying project
MIT notice. They include numeric parameters and optimizer/adaptive/RNG-state
observations from small heads initialized and trained locally by this project.
They contain no third-party released/pretrained weights, SDK binaries or native
assets. Reading them in the inspector does not reconstruct or execute models.
Original provenance, charges, failures and limitations are retained in their
resource README; this grant does not replace separately installed library terms.

## Separately installed components

The project's sole runtime dependency is NumPy. Its distribution includes its own
license and bundled binary component notices; retain those when redistributing
NumPy binaries. The optional native environment is installed separately. The
following describes the observed environment, not a complete dependency license
inventory or a new installation lock:

| Component | Observed version | License evidence and limits |
| --- | --- | --- |
| NumPy | 1.26.4 native; 2.4.6 accepted Windows CPU replay | BSD-3-Clause project code plus bundled component notices in the installed distribution. |
| ManiSkill | 3.0.1 | Installed code LICENSE is Apache-2.0. Installed LICENSE-3RD-PARTY contains a PyTorch3D BSD notice; it does not establish Panda/table asset grants. |
| SAPIEN | 3.0.3 | Installed LICENSE and upstream versioned LICENSE say Apache-2.0; installed metadata says MIT. Preserve this discrepancy rather than relying on the metadata field alone. |
| Gymnasium | 1.2.3 | Installed metadata identifies MIT; consult its shipped license. |
| MiniGrid | 3.1.0 | Installed metadata says MIT License, while its shipped `minigrid-3.1.0.dist-info/licenses/LICENSE` begins with Apache License2.0. Both observations are preserved; this table does not resolve the discrepancy. |
| pygame-ce | 2.5.7 | Installed metadata identifies LGPLv2.1; retain the distribution's bundled notices. |
| cloudpickle | 3.1.2 | Installed metadata identifies BSD-3-Clause and includes a LICENSE; retain its notices. |
| Farama-Notifications | 0.0.6 | Metadata has no license field; the shipped LICENSE begins MIT License. Missing metadata does not erase the shipped notice. |
| Torch | 2.14.0+cpu | Installed metadata declares Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, BSL-1.0 and MIT components. Retain the distribution's notices. |

Primary sources: [NumPy licensing](https://numpy.org/doc/stable/license.html),
[ManiSkill code license](https://github.com/mani-skill/ManiSkill/blob/main/LICENSE),
[ManiSkill third-party notice](https://github.com/mani-skill/ManiSkill/blob/main/LICENSE-3RD-PARTY),
[SAPIEN 3.0.3 license](https://github.com/haosulab/SAPIEN/blob/3.0.3/LICENSE),
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium),
[PyTorch](https://github.com/pytorch/pytorch).
Current upstream pages are supporting context; the installed files establish
the local observations. Missing metadata is not evidence that a component lacks
a license.

## Optional pretrained Choice smoke

The separate local Open-Jev cache contains the pinned source under MIT and
base/adapter/head weights under their accompanying Apache-2.0 terms. Neither
weights nor inference libraries are included in the lightweight demo package.
Exact component identities, original notices and data limitations are recorded
in [the source contract](docs/upstream/open-jev.md) and
[the smoke protocol](artifacts/runs/open-jev-native-smoke-20261022/PROTOCOL.md).
Transformers/PEFT/Accelerate and their transitive installed distributions keep
their own notices. This is not a complete dependency redistribution clearance.

## Native assets: unresolved redistribution scope

The accepted native path selects Panda `panda_v2.urdf`, its referenced meshes and
the table `utils/scene_builder/table/assets/table.glb` from the installed ManiSkill
package. The task cubes are procedural primitives. These assets are not vendored
in this project's wheel or source archive.
The ground builder also references a conditional `grid_texture.png` when
`scene.can_render()`; its individual grant remains unverified. The accepted
factories request no renderer, and this metadata inspection did not execute that
branch or read binary geometry/textures.

The [ManiSkill README license section](https://github.com/mani-skill/ManiSkill#license)
distinguishes permissively licensed rigid-body environment code from assets
under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
That license includes attribution and noncommercial conditions; consult its
[legal text](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en).
An Apache code license or a different Franka repository's license does not prove
the grant for these exact installed meshes/table. Individual asset provenance,
required attribution and any more specific permissions remain unresolved.
Do not interpret the project MIT grant as clearing native asset redistribution
or commercial use. No native assets are redistributed by this change.

Why this: code metadata alone hid both an actual SAPIEN license discrepancy and
the separate ManiSkill asset terms. Keeping the distinctions explicit preserves
the useful CPU example without overstating native release readiness.

## Derived optional CLM head factory

The `make_clm_head` architecture in `src/adapters/clm_decision.py` derives from
Contrastive-LM/CLM `src/clm/heads.py`, revision
`d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6`:
https://github.com/Contrastive-LM/CLM/blob/d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6/src/clm/heads.py
Its original Apache-2.0 grant is retained below; the project MIT statement does
not replace that source's terms. Local changes add argument validation and typed
wrapping. Released encoder/head weights are cached separately and are not bundled
in the lightweight package. The pinned source and data limitations remain in
`docs/upstream/contrastive-lm.md`. No broader redistribution clearance is claimed.

### Original CLM source LICENSE

```text

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

```

## MiniGrid affine TRAIN example

`examples/minigrid_affine_train/episodes.json` contains project-generated synthetic
TRAIN trajectories under the project MIT license; no external dataset, fitted
model, probe or final labels. MiniGrid/Gymnasium retain their existing separate
software licenses. Provenance: examples/minigrid_affine_train/README.md.
