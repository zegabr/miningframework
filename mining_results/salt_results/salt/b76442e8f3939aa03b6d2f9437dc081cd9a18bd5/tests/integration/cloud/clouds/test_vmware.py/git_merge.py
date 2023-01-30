"""
    :codeauthor: Megan Wilhite <mwilhite@saltstack.com>
"""
<<<<<<< /home/ze/Desktop/mining_results/salt_results/salt/b76442e8f3939aa03b6d2f9437dc081cd9a18bd5/tests/integration/cloud/clouds/test_vmware.py/left.py

import socket

||||||| /home/ze/Desktop/mining_results/salt_results/salt/b76442e8f3939aa03b6d2f9437dc081cd9a18bd5/tests/integration/cloud/clouds/test_vmware.py/base.py

# Import Python Libs
from __future__ import absolute_import, print_function, unicode_literals

# Import Salt Libs
from salt.ext import six

# Create the cloud instance name to be used throughout the tests
=======
# Create the cloud instance name to be used throughout the tests
>>>>>>> /home/ze/Desktop/mining_results/salt_results/salt/b76442e8f3939aa03b6d2f9437dc081cd9a18bd5/tests/integration/cloud/clouds/test_vmware.py/right.py
from tests.integration.cloud.helpers.cloud_test_base import TIMEOUT, CloudTest


class VMWareTest(CloudTest):
    """
    Integration tests for the vmware cloud provider in Salt-Cloud
    """

    PROVIDER = "vmware"
    REQUIRED_PROVIDER_CONFIG_ITEMS = ("password", "user", "url")

    def setUp(self):
        super().setUp()
        test_host = self.provider_config["url"]
        try:
            socket.gethostbyname_ex(test_host)
        except OSError as exc:
            self.skipTest(
                "The required vmware host at {} is not available: {}".format(
                    test_host, exc
                )
            )

    def test_instance(self):
        """
        Tests creating and deleting an instance on vmware and installing salt
        """
        # create the instance
        disk_datastore = self.config["vmware-test"]["devices"]["disk"]["Hard disk 2"][
            "datastore"
        ]

        ret_val = self.run_cloud(
            "-p vmware-test {}".format(self.instance_name), timeout=TIMEOUT
        )
        disk_datastore_str = "                [{}] {}/Hard disk 2-flat.vmdk".format(
            disk_datastore, self.instance_name
        )

        # check if instance returned with salt installed
        self.assertInstanceExists(ret_val)
        self.assertIn(
            disk_datastore_str,
            ret_val,
            msg="Hard Disk 2 did not use the Datastore {} ".format(disk_datastore),
        )

        self.assertDestroyInstance()

    def test_snapshot(self):
        """
        Tests creating snapshot and creating vm with --no-deploy
        """
        # create the instance
        ret_val = self.run_cloud(
            "-p vmware-test {} --no-deploy".format(self.instance_name), timeout=TIMEOUT
        )

        # check if instance returned with salt installed
        self.assertInstanceExists(ret_val)

        create_snapshot = self.run_cloud(
<<<<<<< /home/ze/Desktop/mining_results/salt_results/salt/b76442e8f3939aa03b6d2f9437dc081cd9a18bd5/tests/integration/cloud/clouds/test_vmware.py/left.py
            "-a create_snapshot {} snapshot_name='Test Cloud' memdump=True -y".format(
||||||| /home/ze/Desktop/mining_results/salt_results/salt/b76442e8f3939aa03b6d2f9437dc081cd9a18bd5/tests/integration/cloud/clouds/test_vmware.py/base.py
            "-a create_snapshot {0} \
                                         snapshot_name='Test Cloud' \
                                         memdump=True -y".format(
=======
            "-a create_snapshot {} \
                                         snapshot_name='Test Cloud' \
                                         memdump=True -y".format(
>>>>>>> /home/ze/Desktop/mining_results/salt_results/salt/b76442e8f3939aa03b6d2f9437dc081cd9a18bd5/tests/integration/cloud/clouds/test_vmware.py/right.py
                self.instance_name
            ),
            timeout=TIMEOUT,
        )
        s_ret_str = "Snapshot created successfully"

        self.assertIn(s_ret_str, str(create_snapshot))

        self.assertDestroyInstance()

    def test_verify_ssl_false(self):
        """
        Tests creating and deleting an instance on vmware when using
        verify_ssl: False
        """
        profile_name = "vmware_verify_ssl"
        self.add_profile_config(
            "vmware-test", {"verify_ssl": False}, "vmware.conf", profile_name
        )
        # create the instance
        ret_val = self.run_cloud(
            "-p {} {}".format(profile_name, self.instance_name), timeout=TIMEOUT
        )
        # check if instance returned with salt installed
        self.assertInstanceExists(ret_val)
        self.assertDestroyInstance()
