# DonutStudio Flatpak packaging

This directory contains the Flathub submission files for the DonutStudio standalone app.

The manifest uses Flatpak `extra-data` so the proprietary DonutStudio binary is downloaded
from the official Donuts Delivery server during install:

The `apply_extra` script extracts the AppImage payload by skipping the AppImage
runtime header. For the current `DonutStudio-x86_64.AppImage`, the SquashFS payload
starts at byte offset `944632`, so extraction starts at byte `944633`.

```bash
flatpak install -y flathub org.flatpak.Builder
flatpak remote-add --if-not-exists --user flathub https://dl.flathub.org/repo/flathub.flatpakrepo
flatpak run --command=flathub-build org.flatpak.Builder online.donutsdelivery.DonutStudio.yml
flatpak install --user -y ./repo online.donutsdelivery.DonutStudio
flatpak run online.donutsdelivery.DonutStudio
flatpak run --command=flatpak-builder-lint org.flatpak.Builder manifest online.donutsdelivery.DonutStudio.yml
flatpak run --command=flatpak-builder-lint org.flatpak.Builder repo repo
```

For a Flathub submission, add these files to a branch based on `flathub/flathub:new-pr`
and open a pull request titled `Add online.donutsdelivery.DonutStudio`.

This package is standalone-only. VST3 and CLAP plugin installation remains available
through the Linux installer and AUR package because DAWs scan fixed plugin directories.
